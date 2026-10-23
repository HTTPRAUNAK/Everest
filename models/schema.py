"""
Pydantic v2 models: API contract (Sec 6), audit record (Annex D),
and the shared data shapes that flow between the agent and teammates' code.

RetrievedChunk is the contract with the RAG teammate:
their retriever must return list[RetrievedChunk].
"""
from __future__ import annotations

from datetime import date
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


# ---------- enums ----------
class AnswerType(str, Enum):
    RETRIEVED_FACT = "retrieved_fact"
    CALCULATED = "calculated"
    NOT_FOUND = "not_found"
    CLARIFICATION_NEEDED = "clarification_needed"
    REFUSED = "refused"
    CONFLICT_FLAGGED = "conflict_flagged"


class Intent(str, Enum):
    POLICY_FACT = "policy_fact"
    PROCEDURE = "procedure"
    PERSONAL_DATA = "personal_data"
    PERSONAL_ELIGIBILITY = "personal_eligibility"
    WHAT_IF = "what_if"
    AMBIGUOUS = "ambiguous"
    OUT_OF_SCOPE = "out_of_scope"


# ---------- request ----------
class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    as_of_date: Optional[date] = None  # defaults to today in the API layer


# ---------- response building blocks ----------
class Citation(BaseModel):
    doc_id: str
    title: str
    section: Optional[str] = None
    page: Optional[int] = None
    version: Optional[str] = None
    effective_from: Optional[date] = None


class ToolCall(BaseModel):
    tool: str
    input: dict[str, Any] = Field(default_factory=dict)
    output: dict[str, Any] = Field(default_factory=dict)
    status: str = "ok"  # ok | error  (audit only, harmless extra in response)
    ms: int = 0


class AppliedRule(BaseModel):
    rule_id: str
    value: str  # e.g. ">= 75%"
    source_doc_id: str
    source_section: Optional[str] = None


class Conflict(BaseModel):
    doc_ids: list[str]  # the documents / clauses in disagreement
    resolved: bool
    winner: Optional[str] = None  # doc_id that prevailed, if resolved
    step: Optional[str] = None  # "scope" | "supersession" | "authority" | "recency" | "unresolved"
    note: str = ""  # human-readable explanation


# ---------- the /ask response (Sec 6.1) ----------
class AskResponse(BaseModel):
    trace_id: str
    answer: str
    answer_type: AnswerType
    citations: list[Citation] = Field(default_factory=list)
    tools_invoked: list[ToolCall] = Field(default_factory=list)
    applied_rules: list[AppliedRule] = Field(default_factory=list)
    conflicts_detected: list[Conflict] = Field(default_factory=list)
    explanation: str = ""
    as_of_date: date


# ---------- shared with the RAG teammate ----------
class RetrievedChunk(BaseModel):
    text: str
    score: float
    # Source Register metadata (Annex B), stored on each chunk at ingest time
    doc_id: str
    title: str
    section: Optional[str] = None
    page: Optional[int] = None
    version: Optional[str] = None
    doc_type: Optional[str] = None
    authority_level: int = 4  # 1 (highest) .. 5 (untrusted)
    effective_from: Optional[date] = None
    effective_to: Optional[date] = None
    supersedes: list[str] = Field(default_factory=list)  # ["ACAD-REG-2024#7.2", ...]
    scope_programmes: str = "ALL"
    scope_batches: str = "ALL"


# ---------- planning (classify -> tools) ----------
class PlannedTool(BaseModel):
    tool: str
    args: dict[str, Any] = Field(default_factory=dict)


# ---------- audit record (Annex D) ----------
class AuditRecord(BaseModel):
    trace_id: str
    timestamp: str  # ISO-8601 UTC
    student_id: Optional[str] = None
    question_category: Optional[str] = None
    sources_retrieved: list[dict[str, Any]] = Field(default_factory=list)  # doc_id, section, score
    precedence_decision: Optional[str] = None
    conflicts_detected: list[Conflict] = Field(default_factory=list)
    tools_invoked: list[ToolCall] = Field(default_factory=list)
    applied_rules: list[AppliedRule] = Field(default_factory=list)
    answer_type: AnswerType
    model: str = ""
    llm_calls: int = 0
    tokens: int = 0
    latency_ms: int = 0