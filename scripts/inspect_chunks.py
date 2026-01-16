"""CLI to inspect PDF chunking and headings."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure repo root is on PYTHONPATH when running as a script.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from contractflow.core.chunking import chunk_pdf


def _truncate(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    if max_chars <= 3:
        return text[:max_chars]
    return text[: max_chars - 3].rstrip() + "..."


def main() -> None:
    parser = argparse.ArgumentParser(description="Print chunked sections with headings for a PDF.")
    parser.add_argument("pdf_path", type=Path, help="Path to the contract PDF")
    parser.add_argument(
        "--max-chars",
        type=int,
        default=400,
        help="Max chars per chunk to print (0 for full text)",
    )
    parser.add_argument(
        "--no-text",
        action="store_true",
        help="Only print headings and metadata",
    )
    args = parser.parse_args()

    chunks = chunk_pdf(args.pdf_path)
    if not chunks:
        print("No text extracted. Is this a scanned PDF? Use OCR.", file=sys.stderr)
        raise SystemExit(1)

    for chunk in chunks:
        heading = chunk.heading or "none"
        print(f"{chunk.chunk_id} page={chunk.page_num} heading={heading}")
        if not args.no_text:
            text = _truncate(chunk.chunk_text, args.max_chars).strip()
            if text:
                print(text)
        print()


if __name__ == "__main__":
    main()
