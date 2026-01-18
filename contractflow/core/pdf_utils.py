"""Utilities for reading contract PDFs into plain text."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from pypdf import PdfReader


def read_pdf_pages(
    pdf_path: str | Path,
    *,
    use_ocr: bool = False,
    ocr_min_chars: int = 40,
    ocr_lang: str = "eng",
    ocr_dpi: int = 200,
) -> List[str]:
    """Return a list of per-page text extracted from a PDF."""
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")

    reader = PdfReader(str(path))
    pages_text: List[str] = []

    for page in reader.pages:
        text = page.extract_text() or ""
        pages_text.append(text.strip())

    if not use_ocr:
        return pages_text

    if _needs_ocr(pages_text, min_chars=ocr_min_chars):
        return read_pdf_pages_ocr(path, lang=ocr_lang, dpi=ocr_dpi)

    return pages_text


def read_pdf_pages_ocr(
    pdf_path: str | Path,
    *,
    lang: str = "eng",
    dpi: int = 200,
    max_pages: Optional[int] = None,
) -> List[str]:
    """OCR a PDF into per-page text."""
    try:
        from pdf2image import convert_from_path
    except Exception as exc:  # pragma: no cover - depends on optional dependency
        raise RuntimeError(
            "OCR requires pdf2image. Install with: pip install pdf2image"
        ) from exc

    try:
        import pytesseract
    except Exception as exc:  # pragma: no cover - depends on optional dependency
        raise RuntimeError(
            "OCR requires pytesseract. Install with: pip install pytesseract"
        ) from exc

    path = Path(pdf_path)
    images = convert_from_path(str(path), dpi=dpi)
    if max_pages is not None:
        images = images[: max(0, max_pages)]

    pages_text: List[str] = []
    for image in images:
        text = pytesseract.image_to_string(image, lang=lang) or ""
        pages_text.append(text.strip())

    return pages_text


def read_pdf_text(
    pdf_path: str | Path,
    *,
    use_ocr: bool = False,
    ocr_min_chars: int = 40,
    ocr_lang: str = "eng",
    ocr_dpi: int = 200,
) -> str:
    """Return concatenated text from all pages of a PDF."""
    pages_text = read_pdf_pages(
        pdf_path,
        use_ocr=use_ocr,
        ocr_min_chars=ocr_min_chars,
        ocr_lang=ocr_lang,
        ocr_dpi=ocr_dpi,
    )
    return "\n\n".join(pages_text).strip()


def _needs_ocr(pages_text: List[str], *, min_chars: int) -> bool:
    if not pages_text:
        return True
    total_visible = sum(_visible_chars(text) for text in pages_text)
    avg_visible = total_visible / max(1, len(pages_text))
    return avg_visible < max(min_chars, 1)


def _visible_chars(text: str) -> int:
    return sum(1 for ch in text if not ch.isspace())
