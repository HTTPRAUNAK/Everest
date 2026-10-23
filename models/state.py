"""Shared LangGraph state. Every node reads from it and returns a partial update."""
from __future__ import annotations

from datetime import date
from typing import Optional, TypedDict

from models.schema import (
    AnswerType, AppliedRule, Citation, Conflict, Intent,
    PlannedTool, RetrievedChunk, ToolCall,
)


class AgentState(TypedDict, total=False):
    # --- input (set by the API layer) ---
    trace_id: str
    question: str
    student_id: Optional[str]      # from X-Student-Id header ONLY (R7)
    as_of_date: date
    started_at: float              # time.perf_counter() at request start
    extractive_answer: bool

    # --- 1. guard ---
    refusal_reason: Optional[str]  # set => route straight to audit

    # --- 2. classify / route ---
    intent: Intent
    tool_plan: list[PlannedTool]
    clarification: Optional[str]   # question to ask back (ambiguous)

    # --- 3. retrieve / 4. precedence ---
    chunks: list[RetrievedChunk]           # raw retrieval
    valid_chunks: list[RetrievedChunk]     # after precedence filtering
    retrieval_error: Optional[str]
    conflicts: list[Conflict]
    upcoming_changes: list[str]            # not-yet-effective docs worth mentioning
    precedence_note: Optional[str]

    # --- 5. tools ---
    tool_calls: list[ToolCall]
    applied_rules: list[AppliedRule]

    # --- 6. synthesize ---
    answer: str
    explanation: str
    answer_type: AnswerType
    citations: list[Citation]

    # --- bookkeeping for audit (7) ---
    model: str
    llm_calls: int
    tokens: int
