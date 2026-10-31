from collections import deque
from pathlib import Path
from typing import Any

import pytest

from app.agent.graph import agent_graph
from app.agent.nodes import InputTooLong, initialize_state
from app.config import Settings
from app.llm.cassette import CassetteStore
from app.llm.client import LLMClient
from app.llm.types import LLMResult, ToolCall
from app.retrieval.base import Hit


class FakeBackend:
    def __init__(self, results: list[LLMResult]) -> None:
        self.results = deque(results)
        self.calls: list[dict[str, Any]] = []

    def generate(self, **kwargs: Any) -> LLMResult:
        self.calls.append(kwargs)
        return self.results.popleft()

    def embed(self, **kwargs: Any) -> Any:
        raise AssertionError("This agent test must not make embedding requests.")


class FakeRetriever:
    def __init__(self) -> None:
        self.queries: list[str] = []

    def search(self, query: str, top_k: int) -> list[Hit]:
        self.queries.append(query)
        return [
            Hit(
                doc_id="etf-basics",
                title="ETF basics",
                snippet="A fund that holds a basket of assets.",
                score=0.9,
            )
        ][:top_k]


def settings() -> Settings:
    return Settings(
        _env_file=None,
        gemini_model="gemini-test",
        gemini_embed_model="gemini-embed",
        max_input_chars=100,
    )


def result(
    *,
    text: str = "",
    tool_calls: list[ToolCall] | None = None,
) -> LLMResult:
    return LLMResult(text=text, tool_calls=tool_calls or [])


def invoke(
    responses: list[LLMResult],
    message: str,
    retriever: FakeRetriever | None = None,
) -> tuple[dict[str, Any], FakeBackend, FakeRetriever]:
    backend = FakeBackend(responses)
    client = LLMClient(
        settings(),
        backend=backend,
        cassette_store=CassetteStore(Path(".")),
    )
    selected_retriever = retriever or FakeRetriever()
    state = initialize_state(
        message=message,
        llm_client=client,
        retriever_factory=lambda: selected_retriever,
        settings=settings(),
        history=[],
        trace_id="trace-test",
    )
    return agent_graph.invoke(state), backend, selected_retriever


def test_rule_refusal_for_buy_question_uses_zero_llm_calls() -> None:
    result_state, backend, _ = invoke([], "Should I buy X?")

    assert result_state["decision"] == "refused"
    assert "can't recommend" in result_state["answer"]
    assert result_state["disclaimer"].startswith("Fictional demo data.")
    assert backend.calls == []


def test_account_specific_request_escalates_without_model_call() -> None:
    result_state, backend, _ = invoke([], "Please check my account balance.")

    assert result_state["decision"] == "escalated"
    assert "DEMO-" in result_state["answer"]
    assert "no human was contacted" in result_state["answer"]
    assert backend.calls == []


def test_answered_tool_call_is_traced_and_cited() -> None:
    result_state, backend, retriever = invoke(
        [
            result(text='{"intent":"ok","rationale":"General education."}'),
            result(
                tool_calls=[
                    ToolCall(
                        name="search_knowledge",
                        arguments={"query": "ETF basics", "top_k": 2},
                    )
                ]
            ),
            result(text="An ETF pools assets in a fund. [etf-basics]"),
        ],
        "What is an ETF?",
    )

    assert result_state["decision"] == "answered"
    assert result_state["citations"] == ["etf-basics"]
    assert result_state["retrieved_doc_ids"] == ["etf-basics"]
    assert result_state["tool_calls"][0]["name"] == "search_knowledge"
    assert retriever.queries == ["ETF basics"]
    assert len(backend.calls) == 3


def test_unsupported_answer_becomes_no_answer() -> None:
    result_state, _, _ = invoke(
        [
            result(text='{"intent":"ok","rationale":"General question."}'),
            result(text="NO_ANSWER: I found nothing relevant."),
        ],
        "Explain an obscure fictional instrument.",
    )

    assert result_state["decision"] == "no_answer"
    assert result_state["answer"].startswith("I found nothing relevant.")


def test_invalid_citation_is_regenerated_once() -> None:
    result_state, backend, _ = invoke(
        [
            result(text='{"intent":"ok","rationale":"General question."}'),
            result(
                tool_calls=[
                    ToolCall(
                        name="search_knowledge",
                        arguments={"query": "ETF", "top_k": 1},
                    )
                ]
            ),
            result(text="ETFs track baskets. [not-retrieved]"),
            result(text="ETFs track baskets. [etf-basics]"),
        ],
        "What is an ETF?",
    )

    assert result_state["decision"] == "answered"
    assert result_state["citations"] == ["etf-basics"]
    assert len(backend.calls) == 4


def test_second_invalid_citation_returns_no_answer() -> None:
    result_state, _, _ = invoke(
        [
            result(text='{"intent":"ok","rationale":"General question."}'),
            result(
                tool_calls=[
                    ToolCall(
                        name="search_knowledge",
                        arguments={"query": "ETF", "top_k": 1},
                    )
                ]
            ),
            result(text="A claim. [unknown]"),
            result(text="Another claim. [unknown]"),
        ],
        "What is an ETF?",
    )

    assert result_state["decision"] == "no_answer"
    assert result_state["citations"] == []


def test_overlong_message_is_rejected_before_llm_call() -> None:
    backend = FakeBackend([])
    client = LLMClient(
        settings(),
        backend=backend,
        cassette_store=CassetteStore(Path(".")),
    )
    state = initialize_state(
        message="x" * 101,
        llm_client=client,
        retriever_factory=FakeRetriever,
        settings=settings(),
    )

    with pytest.raises(InputTooLong, match="100-character"):
        agent_graph.invoke(state)
    assert backend.calls == []


def test_classifier_off_topic_is_refused() -> None:
    result_state, backend, _ = invoke(
        [result(text='{"intent":"off_topic","rationale":"Not finance."}')],
        "Tell me a joke.",
    )

    assert result_state["decision"] == "refused"
    assert len(backend.calls) == 1
