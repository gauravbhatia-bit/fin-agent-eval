"""LangGraph assembly for the guarded chat flow."""

from langgraph.graph import END, START, StateGraph

from app.agent.nodes import (
    AgentState,
    agent_node,
    escalation_node,
    finalize_node,
    grounding_check,
    input_guard,
    refusal_node,
)


def build_agent_graph():
    """Compile the guard → route → answer → grounding → finalize graph."""
    graph = StateGraph(AgentState)
    graph.add_node("input_guard", input_guard)
    graph.add_node("refusal", refusal_node)
    graph.add_node("escalation", escalation_node)
    graph.add_node("agent", agent_node)
    graph.add_node("grounding_check", grounding_check)
    graph.add_node("finalize", finalize_node)
    graph.add_edge(START, "input_guard")
    graph.add_conditional_edges(
        "input_guard",
        lambda state: state["route"],
        {
            "refuse": "refusal",
            "escalate": "escalation",
            "agent": "agent",
        },
    )
    graph.add_edge("refusal", "finalize")
    graph.add_edge("escalation", "finalize")
    graph.add_edge("agent", "grounding_check")
    graph.add_edge("grounding_check", "finalize")
    graph.add_edge("finalize", END)
    return graph.compile()


agent_graph = build_agent_graph()
