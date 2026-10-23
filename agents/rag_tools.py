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

# Retrieve a compact context; the best hit's source page is expanded below so
# tables and lists split across chunks remain complete.
TOP_K = 5


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
    seen_chunks: set[str] = set()
    results = retrieval.search(query, top_k=top_k)
    first = results[0] if results else None
    first_page = (
        (str(first.get("doc_id") or "unknown"), _to_int(first.get("page"), 0))
        if first and _to_int(first.get("page"), 0) > 0
        else None
    )
    page_expanded = False

    for r in results:
        doc_id = str(r.get("doc_id") or "unknown")
        page = _to_int(r.get("page"), 0) or None
        score = float(r.get("score", 0.0))

        page_key = (doc_id, page) if page is not None else None
        if page_key and page_key == first_page and not page_expanded:
            siblings = retrieval.get_page_chunks(doc_id, page)

            def chunk_order(item: dict) -> int:
                match = re.search(r"_c(\d+)$", str(item.get("chunk_id", "")))
                return int(match.group(1)) if match else 0

            siblings.sort(key=chunk_order)
            if not siblings:
                out.append(_to_retrieved_chunk(r, score, doc_id))
                if r.get("chunk_id"):
                    seen_chunks.add(str(r["chunk_id"]))
            for sibling in siblings:
                chunk_id = str(sibling.get("chunk_id") or "")
                text = (sibling.get("text") or "").strip()
                if not text or (chunk_id and chunk_id in seen_chunks):
                    continue
                out.append(_to_retrieved_chunk(sibling, score, doc_id))
                if chunk_id:
                    seen_chunks.add(chunk_id)
            page_expanded = True
            continue

        chunk_id = str(r.get("chunk_id") or "")
        if chunk_id and chunk_id in seen_chunks:
            continue
        out.append(_to_retrieved_chunk(r, score, doc_id))
        if chunk_id:
            seen_chunks.add(chunk_id)
    return out


def _to_retrieved_chunk(r: dict, score: float, doc_id: str) -> RetrievedChunk:
    text = (r.get("text") or "").strip()
    return RetrievedChunk(
        text=text,
        score=score,
        doc_id=doc_id,
        title=str(r.get("title") or doc_id),
        section=str(r["section"]) if r.get("section") else None,
        page=_to_int(r.get("page"), 0) or None,
        version=str(r["version"]) if r.get("version") not in (None, "") else None,
        doc_type=r.get("doc_type") or None,
        authority_level=_to_int(r.get("authority_level"), 4),
        effective_from=_to_date(r.get("effective_from")),
        effective_to=_to_date(r.get("effective_to")),
        supersedes=_to_list(r.get("supersedes")),
        scope_programmes=str(r.get("scope_programmes") or "ALL"),
        scope_batches=str(r.get("scope_batches") or "ALL"),
    )
