
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
EXTRACTED_DIR = DATA_DIR / "extracted"
CHUNKS_DIR = DATA_DIR / "chunks"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150


def split_text(text: str) -> list[str]:
    """Split text into overlapping character-based chunks."""
    text = " ".join(text.split())

    if not text:
        return []

    chunks = []
    start = 0

    while start < len(text):
        end = min(start + CHUNK_SIZE, len(text))
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end == len(text):
            break

        start = end - CHUNK_OVERLAP

    return chunks


def chunk_document(pages: list[dict]) -> list[dict]:
    """Create chunks while preserving the original source and page."""
    chunks = []

    for page_data in pages:
        page_chunks = split_text(page_data.get("text", ""))

        for index, text in enumerate(page_chunks, start=1):
            chunks.append({
                "chunk_id": (
                    f"{Path(page_data['source']).stem}"
                    f"_p{page_data['page']}_c{index}"
                ),
                "source": page_data["source"],
                "page": page_data["page"],
                "doc_id": page_data.get("doc_id") or Path(page_data["source"]).stem,
                "text": text,
                **{
                    key: value
                    for key, value in page_data.items()
                    if key not in {"source", "page", "text"} and value is not None
                },
            })

    return chunks


def main():
    CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(EXTRACTED_DIR.glob("*.json"))

    if not files:
        print(f"No extracted JSON files found in {EXTRACTED_DIR}")
        return

    for file_path in files:
        try:
            with file_path.open("r", encoding="utf-8") as file:
                pages = json.load(file)

            chunks = chunk_document(pages)
            output_path = CHUNKS_DIR / file_path.name

            with output_path.open("w", encoding="utf-8") as file:
                json.dump(chunks, file, ensure_ascii=False, indent=2)

            print(
                f"SUCCESS: {file_path.name} | "
                f"{len(chunks)} chunks created | "
                f"Saved to {output_path}"
            )

        except (OSError, json.JSONDecodeError, KeyError, TypeError) as error:
            print(f"ERROR processing {file_path.name}: {error}")


if __name__ == "__main__":
    main()
