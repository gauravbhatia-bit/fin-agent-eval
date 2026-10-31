"""LangGraph nodes for request guard, tool loop, and grounding checks."""

import json
import re
from collections.abc import Callable
from typing import Any, Protocol, TypedDict
from uuid import uuid4

from app.agent.policy import (
    Intent,
    IntentClassification,
    classify_with_rules,
    load_policy,
    refusal_message,
    route_intent,
)
from app.agent.prompts import (
    GROUNDING_REMINDER,
    GUARD_PROMPT,
    SYSTEM_PROMPT,
)
from app.config import Settings
from app.llm.types import LLMResult
from app.retrieval.base import Retriever
from app.tools.registry import dispatch_tool, tool_schemas

CITATION_PATTERN = re.compile(r"\[([a-z0-9][a-z0-9-]*)\]")
MAX_TOOL_STEPS = 4
NO_ANSWER_PREFIX = "NO_ANSWER:"
NO_ANSWER_TEXT = (
    "I don't have enough verified information in this fictional demo pack to answer."
)


class AgentError(RuntimeError):
    """Raised when an agent response cannot be processed safely."""


class AgentLLM(Protocol):
    """Subset of the LLM client used by the graph."""

    def generate(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
        response_schema: dict[str, Any] | None = None,
        purpose: str = "agent",
        prompt_version: str = "v1",
    ) -> LLMResult: ...


class AgentState(TypedDict, total=False):
    """State carried between LangGraph nodes."""

    message: str
    history: list[dict[str, Any]]
    trace_id: str
    route: str
    intent: Intent
    decision: str
    answer: str
    tool_calls: list[dict[str, Any]]
    retrieved_doc_ids: list[str]
    citations: list[str]
    disclaimer: str
    conversation: list[dict[str, Any]]
    llm_client: AgentLLM
    retriever: Retriever
    settings: Settings
    retriever_factory: Callable[[], Retriever]


class InputTooLong(ValueError):
    """Raised when request text exceeds the configured hard limit."""


def input_guard(state: AgentState) -> dict[str, Any]:
    """Apply hard length and rule checks before invoking the classifier."""
    message = state["message"]
    settings = state["settings"]
    if len(message) > settings.max_input_chars:
        raise InputTooLong(
            f"Message exceeds the {settings.max_input_chars}-character limit."
        )
    rule_match = classify_with_rules(message)
    if rule_match is not None:
        return {
            "intent": rule_match.intent,
            "route": rule_match.route,
        }
    result = state["llm_client"].generate(
        [
            {"role": "system", "content": GUARD_PROMPT},
            {"role": "user", "content": message},
        ],
        temperature=0,
        response_schema=IntentClassification.model_json_schema(),
        purpose="guard",
    )
    if result.tool_calls:
        raise AgentError("Intent classifier returned an unexpected tool call.")
    try:
        classification = IntentClassification.model_validate_json(result.text)
    except ValueError as error:
        raise AgentError(
            "Intent classifier returned invalid structured output."
        ) from error
    route = route_intent(classification.intent)
    result: dict[str, Any] = {
        "intent": classification.intent,
        "route": route,
    }
    if route == "agent":
        result["retriever"] = state["retriever_factory"]()
    return result


def refusal_node(state: AgentState) -> dict[str, Any]:
    """Finish with a refusal and no tool or model work."""
    intent = state["intent"]
    return {
        "decision": "refused",
        "answer": refusal_message(intent),
        "tool_calls": [],
        "retrieved_doc_ids": [],
        "citations": [],
    }


def escalation_node(state: AgentState) -> dict[str, Any]:
    """Create the deterministic local escalation ticket."""
    result = dispatch_tool(
        "escalate_to_human",
        {"reason": state["message"][:500]},
    )
    ticket_id = str(result.get("ticket_id", ""))
    if not ticket_id:
        raise AgentError("Escalation stub did not return a ticket identifier.")
    return {
        "decision": "escalated",
        "answer": (
            "This request needs a person. The fictional local reference is "
            f"{ticket_id}; "
            "no human was contacted."
        ),
        "tool_calls": [
            {
                "name": "escalate_to_human",
                "arguments": {"reason": state["message"][:500]},
                "result": result,
            }
        ],
        "retrieved_doc_ids": [],
        "citations": [],
    }


def agent_node(state: AgentState) -> dict[str, Any]:
    """Run the explicit, bounded function-call loop."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        *state.get("history", []),
        {"role": "user", "content": state["message"]},
    ]
    tools = tool_schemas()
    calls: list[dict[str, Any]] = []
    retrieved_ids: list[str] = []
    final_text = ""
    decision = "answered"
    llm_client = state["llm_client"]

    for _ in range(MAX_TOOL_STEPS):
        response = llm_client.generate(messages, tools=tools, purpose="agent")
        if not response.tool_calls:
            final_text = response.text.strip()
            break
        if len(calls) + len(response.tool_calls) > MAX_TOOL_STEPS:
            final_text = _force_final_answer(llm_client, messages)
            break
        messages.append(
            {
                "role": "assistant",
                "content": response.text,
                "tool_calls": [
                    tool_call.model_dump() for tool_call in response.tool_calls
                ],
            }
        )
        for tool_call in response.tool_calls:
            arguments = tool_call.arguments.copy()
            if tool_call.name == "escalate_to_human":
                reason = arguments.get("reason", state["message"])
                arguments["reason"] = str(reason)[:500]
            result = dispatch_tool(
                tool_call.name,
                arguments,
                retriever=state["retriever"],
            )
            calls.append(
                {
                    "name": tool_call.name,
                    "arguments": arguments,
                    "result": result,
                }
            )
            if tool_call.name == "search_knowledge":
                for hit in result.get("results", []):
                    if isinstance(hit, dict) and isinstance(hit.get("doc_id"), str):
                        retrieved_ids.append(hit["doc_id"])
            messages.append(
                {
                    "role": "tool",
                    "name": tool_call.name,
                    "content": result,
                }
            )
            if tool_call.name == "escalate_to_human" and "error" not in result:
                ticket_id = str(result["ticket_id"])
                return {
                    "decision": "escalated",
                    "answer": (
                        f"This request needs a person. The fictional local reference "
                        f"is {ticket_id}; no human was contacted."
                    ),
                    "tool_calls": calls,
                    "retrieved_doc_ids": list(dict.fromkeys(retrieved_ids)),
                    "citations": [],
                }
    else:
        final_text = _force_final_answer(llm_client, messages)

    if not final_text:
        raise AgentError("Agent returned neither an answer nor a tool call.")
    if final_text.casefold().startswith(NO_ANSWER_PREFIX.casefold()):
        decision = "no_answer"
        final_text = final_text[len(NO_ANSWER_PREFIX) :].strip()
        if not final_text:
            final_text = NO_ANSWER_TEXT
    citations = list(dict.fromkeys(CITATION_PATTERN.findall(final_text)))
    return {
        "decision": decision,
        "answer": final_text,
        "tool_calls": calls,
        "retrieved_doc_ids": list(dict.fromkeys(retrieved_ids)),
        "citations": citations,
        "conversation": messages,
    }


def _force_final_answer(llm_client: AgentLLM, messages: list[dict[str, Any]]) -> str:
    messages.append(
        {
            "role": "system",
            "content": (
                "Tool-call limit reached. Give the best concise answer from the "
                "tool results already present; do not call more tools. If evidence "
                "is insufficient, start with NO_ANSWER:."
            ),
        }
    )
    result = llm_client.generate(messages, tools=None, purpose="agent")
    if result.tool_calls:
        raise AgentError("Agent returned a tool call after the tool-call limit.")
    return result.text.strip()


def grounding_check(state: AgentState) -> dict[str, Any]:
    """Require citations to refer only to documents actually retrieved."""
    if state["decision"] != "answered" or not state.get("retrieved_doc_ids"):
        return {}
    retrieved = set(state["retrieved_doc_ids"])
    citations = list(dict.fromkeys(CITATION_PATTERN.findall(state["answer"])))
    if citations and set(citations).issubset(retrieved):
        return {"citations": citations}

    messages = [
        *state["conversation"],
        {"role": "assistant", "content": state["answer"]},
        {"role": "system", "content": GROUNDING_REMINDER},
        {
            "role": "assistant",
            "content": json.dumps({"retrieved_doc_ids": sorted(retrieved)}),
        },
    ]
    result = state["llm_client"].generate(
        messages,
        tools=None,
        purpose="agent",
    )
    if result.tool_calls:
        raise AgentError("Grounding regeneration returned an unexpected tool call.")
    answer = result.text.strip()
    if answer.casefold().startswith(NO_ANSWER_PREFIX.casefold()):
        answer = answer[len(NO_ANSWER_PREFIX) :].strip() or NO_ANSWER_TEXT
        return {
            "decision": "no_answer",
            "answer": answer,
            "citations": [],
        }
    citations = list(dict.fromkeys(CITATION_PATTERN.findall(answer)))
    if not citations or not set(citations).issubset(retrieved):
        return {
            "decision": "no_answer",
            "answer": NO_ANSWER_TEXT,
            "citations": [],
        }
    return {"answer": answer, "citations": citations}


def finalize_node(state: AgentState) -> dict[str, Any]:
    """Attach the configured disclaimer exactly once."""
    disclaimer = load_policy().disclaimer
    answer = state["answer"].strip()
    if disclaimer.casefold() not in answer.casefold():
        answer = f"{answer}\n\n{disclaimer}"
    return {"answer": answer, "disclaimer": disclaimer}


def initialize_state(
    message: str,
    llm_client: AgentLLM,
    retriever_factory: Callable[[], Retriever],
    settings: Settings,
    history: list[dict[str, Any]] | None = None,
    trace_id: str | None = None,
) -> AgentState:
    """Create graph input with a trace identifier and empty output collections."""
    return {
        "message": message,
        "history": history or [],
        "trace_id": trace_id or uuid4().hex,
        "llm_client": llm_client,
        "retriever_factory": retriever_factory,
        "settings": settings,
        "tool_calls": [],
        "retrieved_doc_ids": [],
        "citations": [],
    }
