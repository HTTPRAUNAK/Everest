"""Minimal local POST /ask endpoint for exercising the agent graph."""
from __future__ import annotations

import os
from time import perf_counter

from fastapi import FastAPI, Header
from fastapi.middleware.cors import CORSMiddleware

from agents.agent import classify_node
from agents.graph import graph
from models.schema import AnswerType, AskRequest, AskResponse, Intent
from utils.utils import new_trace_id, today


app = FastAPI(
    title="Everest Student Services API",
    description="Ask university questions and receive grounded answers, citations, and tool results.",
    version="0.1.0",
)

cors_origins = [
    origin.strip()
    for origin in os.getenv("CORS_ORIGINS", "*").split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "X-Student-Id"],
)


@app.get("/health")
def health() -> dict[str, str]:
    """Basic liveness check for the local FastAPI process."""
    return {"status": "ok"}


@app.post("/route")
def route(request: AskRequest) -> dict[str, str | bool]:
    """Expose the agent router so the UI can request identity only when needed."""
    classification = classify_node({"question": request.question})
    intent = classification["intent"]
    requires_student_id = intent in {
        Intent.PERSONAL_DATA,
        Intent.PERSONAL_ELIGIBILITY,
        Intent.WHAT_IF,
    }
    return {
        "intent": intent.value,
        "requires_student_id": requires_student_id,
    }


@app.post(
    "/ask",
    response_model=AskResponse,
    summary="Ask a university question",
    description=(
        "Send the question as JSON. For personal questions, pass the authenticated "
        "student ID in the X-Student-Id header; do not put identity in the request body."
    ),
)
def ask(
    request: AskRequest,
    x_student_id: str | None = Header(
        default=None,
        alias="X-Student-Id",
        description="Authenticated student ID; optional for general policy questions.",
    ),
) -> AskResponse:
    as_of_date = request.as_of_date or today()
    result = graph.invoke(
        {
            "trace_id": new_trace_id(),
            "question": request.question,
            "student_id": x_student_id,
            "as_of_date": as_of_date,
            "started_at": perf_counter(),
        }
    )
    return AskResponse(
        trace_id=result["trace_id"],
        answer=result.get("answer", ""),
        answer_type=result.get("answer_type", AnswerType.NOT_FOUND),
        citations=result.get("citations", []),
        tools_invoked=result.get("tool_calls", []),
        applied_rules=result.get("applied_rules", []),
        conflicts_detected=result.get("conflicts", []),
        explanation=result.get("explanation", ""),
        as_of_date=as_of_date,
    )
