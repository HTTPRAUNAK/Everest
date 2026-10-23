"""
Adapter: teammate's app/retrieval.py  ->  list[RetrievedChunk].

Used by the Retrieve node (node 3). This is NOT an LLM-planned tool; retrieval
always runs for policy/procedure questions. Their search() returns flat dicts
(Chroma metadata + text + score); we normalise types here so the precedence node
can trust the fields (dates parsed, lists parsed, ints coerced).
"""
from __future__ import annotations

import json
import re
from datetime import date
from typing import Any, Optional

from utils import retrieval
from models.schema import RetrievedChunk

# Their default is 3. Precedence needs to SEE competing versions, so pull more.
TOP_K = 8


def _to_date(v: Any) -> Optional[date]:
    if not v:
        return None
    try:
        return date.fromisoformat(str(v).strip()[:10])
    except ValueError:
        return None


def _to_int(v: Any, default: int) -> int:
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def _to_list(v: Any) -> list[str]:
    """Chroma stores lists as JSON strings; Annex B uses 'a;b' strings. Handle both."""
    if not v:
        return []
    if isinstance(v, list):
        return [str(x) for x in v]
    s = str(v).strip()
    if s.startswith("["):
        try:
            return [str(x) for x in json.loads(s)]
        except json.JSONDecodeError:
            pass
    return [x.strip() for x in re.split(r"[;,]", s) if x.strip()]


def retrieve_chunks(query: str, top_k: int = TOP_K) -> list[RetrievedChunk]:
    """Semantic search -> normalised chunks. Raises if the vector store is broken
    (the node decides what to do; never silently return [] on an error, or a
    crash would look like a legitimate 'not_found')."""
    out: list[RetrievedChunk] = []
    for r in retrieval.search(query, top_k=top_k):
        text = (r.get("text") or "").strip()
        if not text:
            continue
        doc_id = str(r.get("doc_id") or "unknown")
        out.append(RetrievedChunk(
            text=text,
            score=float(r.get("score", 0.0)),
            doc_id=doc_id,
            title=str(r.get("title") or doc_id),
            section=str(r["section"]) if r.get("section") else None,
            page=_to_int(r.get("page"), 0) or None,  # their default page is 0
            version=str(r["version"]) if r.get("version") not in (None, "") else None,
            doc_type=r.get("doc_type") or None,
            authority_level=_to_int(r.get("authority_level"), 4),
            effective_from=_to_date(r.get("effective_from")),
            effective_to=_to_date(r.get("effective_to")),
            supersedes=_to_list(r.get("supersedes")),
            scope_programmes=str(r.get("scope_programmes") or "ALL"),
            scope_batches=str(r.get("scope_batches") or "ALL"),
        ))
    return out
