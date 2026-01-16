"""Utilities for reading contract PDFs into plain text."""

from __future__ import annotations

from pathlib import Path
from typing import List

from pypdf import PdfReader


def read_pdf_pages(pdf_path: str | Path) -> List[str]:
    """Return a list of per-page text extracted from a PDF."""
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")

    reader = PdfReader(str(path))
    pages_text: List[str] = []

    for page in reader.pages:
        text = page.extract_text() or ""
        pages_text.append(text.strip())

    return pages_text


def read_pdf_text(pdf_path: str | Path) -> str:
    """Return concatenated text from all pages of a PDF."""
    pages_text = read_pdf_pages(pdf_path)
    return "\n\n".join(pages_text).strip()
