import json
from pathlib import Path

import pymupdf as fitz

# Locate the source PDFs and generated files relative to this Python file.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DOCUMENTS_DIR = DATA_DIR
EXTRACTED_DIR = DATA_DIR / "extracted"


def extract_pdf(pdf_path: Path, source_name: str | None = None) -> list[dict]:
    """Extract text page-by-page, using local Tesseract OCR for empty pages if available."""
    pages = []
    display_name = source_name or pdf_path.name

    with fitz.open(pdf_path) as pdf:
        for page_number, page in enumerate(pdf, start=1):
            text = page.get_text("text").strip()

            # Scanned PDFs often have no selectable text. Try OCR only on those pages.
            if not text:
                try:
                    text_page = page.get_textpage_ocr(
                        language="eng",
                        dpi=200,
                        full=True,
                    )
                    text = page.get_text("text", textpage=text_page).strip()
                except (RuntimeError, ValueError, OSError):
                    # Tesseract may not be installed or configured. Normal text PDFs
                    # should still work, and the upload endpoint will explain if no text remains.
                    text = ""

            if text:
                pages.append({
                    "source": display_name,
                    "page": page_number,
                    "text": text,
                })

    return pages


def ingest_documents():
    """Process all PDFs in the documents folder into extracted JSON files."""
    EXTRACTED_DIR.mkdir(parents=True, exist_ok=True)
    pdf_files = sorted(DOCUMENTS_DIR.glob("*.pdf"))

    if not pdf_files:
        print(f"No PDFs found in {DOCUMENTS_DIR}")
        return

    for pdf_path in pdf_files:
        try:
            pages = extract_pdf(pdf_path)

            if not pages:
                print(
                    f"No text could be extracted from {pdf_path.name}. "
                    "If this is a scanned PDF, install/configure Tesseract OCR."
                )
                continue

            output_path = EXTRACTED_DIR / f"{pdf_path.stem}.json"
            with output_path.open("w", encoding="utf-8") as file:
                json.dump(pages, file, ensure_ascii=False, indent=2)

            print(f"SUCCESS: {pdf_path.name} — {len(pages)} text pages extracted.")
            print(f"Saved to: {output_path}")

        except (OSError, fitz.FileDataError, RuntimeError) as error:
            print(f"ERROR processing {pdf_path.name}: {error}")


if __name__ == "__main__":
    ingest_documents()
