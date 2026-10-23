"""LangGraph assembly for the student-services assistant."""
from __future__ import annotations
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__) , '..')))


from typing import Literal

from langgraph.graph import END, START, StateGraph

from agent import (
    audit_node,
    classify_node,
    guard_node,
    precedence_node,
    retrieve_node,
    synthesize_node,
    tools_node,
)
from models.schema import Intent
from models.state import AgentState


def after_guard(state: AgentState) -> Literal["classify", "audit"]:
    """Refusals go directly to audit, as specified in architecture.md."""
    return "audit" if state.get("refusal_reason") else "classify"


def after_classify(
    state: AgentState,
) -> Literal["retrieve", "tools", "synthesize"]:
    intent = state.get("intent")
    if intent in (Intent.POLICY_FACT, Intent.PROCEDURE):
        return "retrieve"
    if intent in (
        Intent.PERSONAL_DATA,
        Intent.PERSONAL_ELIGIBILITY,
        Intent.WHAT_IF,
    ):
        return "tools"
    return "synthesize"


def build_graph():
    """Build and compile the seven-node workflow described in architecture.md."""
    builder = StateGraph(AgentState)
    builder.add_node("guard", guard_node)
    builder.add_node("classify", classify_node)
    builder.add_node("retrieve", retrieve_node)
    builder.add_node("precedence", precedence_node)
    builder.add_node("tools", tools_node)
    builder.add_node("synthesize", synthesize_node)
    builder.add_node("audit", audit_node)

    builder.add_edge(START, "guard")
    builder.add_conditional_edges(
        "guard",
        after_guard,
        {"classify": "classify", "audit": "audit"},
    )
    builder.add_conditional_edges(
        "classify",
        after_classify,
        {
            "retrieve": "retrieve",
            "tools": "tools",
            "synthesize": "synthesize",
        },
    )
    builder.add_edge("tools", "retrieve")
    builder.add_edge("retrieve", "precedence")
    builder.add_edge("precedence", "synthesize")
    builder.add_edge("synthesize", "audit")
    builder.add_edge("audit", END)
    return builder.compile()


# Ready for API/service code to invoke with an AgentState-compatible mapping.
graph = build_graph()

print(graph.get_graph().draw_mermaid())

# To render it as an image in a notebook:

from IPython.display import Image, display
from agents.graph import graph

img = Image(graph.get_graph().draw_mermaid_png())
with open("graph.png" , "wb") as f:
    f.write(img.data)
