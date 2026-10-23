"""Persistent ChromaDB retrieval for the university policy RAG pipeline.

Documents are embedded with Sentence Transformers. Chroma stores chunk text,
embeddings, and filterable citation/source metadata on disk.
"""
import json
import os
import re
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHUNKS_DIR = PROJECT_ROOT / "data" / "chunks"
CHROMA_DIR = Path(os.getenv("CHROMA_DIR", str(PROJECT_ROOT / "data" / "chroma")))
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION", "university_policy_chunks")
EMBEDDING_MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)
DEFAULT_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "3"))

_client = None
_collection = None
_embedding_model = None


def get_embedding_model():
    """Load the same local embedding model for both documents and questions."""
    global _embedding_model
    if _embedding_model is None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as error:
            raise RuntimeError(
                "Sentence Transformers is not installed. Run: "
                "python -m pip install -r requirements.txt"
            ) from error
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _embedding_model


def get_collection():
    """Open/create the persistent Chroma collection (no re-ingestion on restart)."""
    global _client, _collection
    if _collection is None:
        try:
            import chromadb
        except ImportError as error:
            raise RuntimeError(
                "ChromaDB is not installed. Run: python -m pip install -r requirements.txt"
            ) from error
        CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        _client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        _collection = _client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


def _safe_metadata(item: dict[str, Any]) -> dict[str, Any]:
    """Chroma metadata values must be scalar strings, numbers, or booleans."""
    metadata: dict[str, Any] = {}
    for key, value in item.items():
        if key in {"text", "embedding", "vector"} or value is None:
            continue
        if isinstance(value, (str, int, float, bool)):
            metadata[key] = value
        elif isinstance(value, (list, tuple, dict)):
            metadata[key] = json.dumps(value, ensure_ascii=False, sort_keys=True)
    metadata.setdefault("source", "unknown")
    metadata.setdefault("page", 0)
    metadata.setdefault("doc_id", Path(str(metadata["source"])).stem)
    metadata.setdefault("chunk_id", "")
    return metadata


def load_chunks() -> list[dict]:
    """Read chunks produced by app.chunking."""
    chunks: list[dict] = []
    for file_path in sorted(CHUNKS_DIR.glob("*.json")):
        with file_path.open("r", encoding="utf-8") as file:
            items = json.load(file)
        if not isinstance(items, list):
            print(f"Skipping invalid chunk file: {file_path.name}")
            continue
        for item in items:
            if isinstance(item, dict) and str(item.get("text", "")).strip():
                item = dict(item)
                item.setdefault("doc_id", file_path.stem)
                item.setdefault("chunk_id", f"{file_path.stem}_{len(chunks)}")
                chunks.append(item)
    return chunks


def upsert_chunks(chunks: list[dict], batch_size: int = 32) -> int:
    """Upsert chunks by stable chunk ID; re-ingesting a doc updates, not duplicates."""
    if not chunks:
        return 0
    collection = get_collection()
    model = get_embedding_model()
    indexed = 0
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start:start + batch_size]
        texts = [str(item["text"]) for item in batch]
        vectors = model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        ).tolist()
        ids = []
        metadatas = []
        for i, item in enumerate(batch):
            metadata = _safe_metadata(item)
            chunk_id = str(item.get("chunk_id") or f"{metadata['doc_id']}_{start+i}")
            metadata["chunk_id"] = chunk_id
            ids.append(chunk_id)
            metadatas.append(metadata)
        collection.upsert(
            ids=ids,
            documents=texts,
            embeddings=vectors,
            metadatas=metadatas,
        )
        indexed += len(batch)
    return indexed


def upsert_document(doc_id: str, chunks: list[dict]) -> int:
    """Replace all chunks for one document id, keeping ingestion idempotent."""
    collection = get_collection()
    existing = collection.get(where={"doc_id": doc_id}, include=["metadatas"])
    existing_ids = existing.get("ids", [])
    normalized = []
    for index, chunk in enumerate(chunks):
        item = dict(chunk)
        item["doc_id"] = doc_id
        item.setdefault("chunk_id", f"{doc_id}_chunk_{index + 1}")
        normalized.append(item)
    new_ids = {str(item["chunk_id"]) for item in normalized}
    stale_ids = [chunk_id for chunk_id in existing_ids if chunk_id not in new_ids]
    count = upsert_chunks(normalized)
    if stale_ids:
        collection.delete(ids=stale_ids)
    return count


def build_index() -> int:
    """Index all local chunk JSON files into persistent ChromaDB."""
    chunks = load_chunks()
    if not chunks:
        raise RuntimeError("No chunks found. Run document extraction and chunking first.")
    # Group by doc_id so a changed document replaces stale chunks too.
    grouped: dict[str, list[dict]] = {}
    for chunk in chunks:
        doc_id = str(chunk.get("doc_id") or Path(str(chunk.get("source", "unknown"))).stem)
        grouped.setdefault(doc_id, []).append(chunk)
    total = 0
    for doc_id, document_chunks in grouped.items():
        total += upsert_document(doc_id, document_chunks)
        print(f"Indexed {len(document_chunks)} chunks for {doc_id}")
    print(f"ChromaDB path: {CHROMA_DIR}")
    print(f"Total indexed chunks: {total}")
    return total


def _query_filter(where: dict[str, Any] | None) -> dict[str, Any] | None:
    """Pass through Chroma-compatible metadata filters."""
    if not where:
        return None
    # Chroma requires a single operator per level; callers should use valid Chroma filters.
    return where


def search(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    where: dict[str, Any] | None = None,
    doc_id: str | None = None,
) -> list[dict]:
    """Retrieve ranked chunks with citation metadata for downstream precedence/synthesis.

    `where` accepts Chroma metadata filters, e.g. {"programme": "B.Tech CSE"}.
    `doc_id` is a convenience filter. The returned `score` is cosine similarity
    (1 - Chroma cosine distance), not a probability.
    """
    query = query.strip()
    if not query or top_k <= 0:
        return []
    collection = get_collection()
    if collection.count() == 0:
        return []
    query_vector = get_embedding_model().encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    ).tolist()[0]
    filters = _query_filter(where)
    if doc_id:
        doc_filter = {"doc_id": doc_id}
        filters = {"$and": [filters, doc_filter]} if filters else doc_filter

    result = collection.query(
        query_embeddings=[query_vector],
        n_results=min(top_k, collection.count()),
        where=filters,
        include=["documents", "metadatas", "distances"],
    )
    documents = (result.get("documents") or [[]])[0]
    metadatas = (result.get("metadatas") or [[]])[0]
    distances = (result.get("distances") or [[]])[0]
    ids = (result.get("ids") or [[]])[0]
    passages = []
    for chunk_id, text, metadata, distance in zip(ids, documents, metadatas, distances):
        metadata = metadata or {}
        # Keep legacy fields expected by main.py and expose richer source metadata.
        passages.append({
            **metadata,
            "chunk_id": metadata.get("chunk_id", chunk_id),
            "source": metadata.get("source", "unknown"),
            "page": metadata.get("page", 0),
            "text": text or "",
            "score": max(-1.0, min(1.0, 1.0 - float(distance))),
        })
    return passages


def get_page_chunks(doc_id: str, page: int) -> list[dict]:
    """Return every indexed chunk on a source page to restore split tables/lists."""
    collection = get_collection()
    result = collection.get(
        where={"$and": [{"doc_id": doc_id}, {"page": page}]},
        include=["documents", "metadatas"],
    )
    ids = result.get("ids", [])
    documents = result.get("documents", [])
    metadatas = result.get("metadatas", [])
    chunks = []
    for chunk_id, text, metadata in zip(ids, documents, metadatas):
        metadata = metadata or {}
        chunks.append({
            **metadata,
            "chunk_id": metadata.get("chunk_id", chunk_id),
            "text": text or "",
        })
    return chunks


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "--build":
        try:
            build_index()
        except Exception as error:
            print(f"Could not build ChromaDB index: {error}")
            print("Check installed dependencies and that the embedding model can load.")
        return
    question = " ".join(sys.argv[1:]).strip() or input(
        "Ask a question about the university documents: "
    )
    try:
        results = search(question)
        if not results:
            print("No indexed passages found. Run: python -m app.retrieval --build")
            return
        print("\\nRetrieved passages:\\n")
        for number, result in enumerate(results, start=1):
            print(f"--- Result {number} | Cosine similarity: {result['score']:.3f} ---")
            print(f"Source: {result['source']}, page {result['page']}")
            print(result["text"])
            print()
    except Exception as error:
        print(f"Retrieval failed: {error}")


if __name__ == "__main__":
    main()
