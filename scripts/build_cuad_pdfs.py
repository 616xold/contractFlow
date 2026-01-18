"""Generate PDFs from the public CUAD dataset text."""

from __future__ import annotations

import argparse
import json
import textwrap
import urllib.request
import zipfile
from pathlib import Path


CUAD_ZIP_URL = "https://github.com/TheAtticusProject/cuad/raw/main/data.zip"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate PDFs from CUAD dataset text.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/raw_pdfs"),
        help="Output directory for generated PDFs",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=25,
        help="Number of CUAD documents to convert (default: 25)",
    )
    parser.add_argument(
        "--line-width",
        type=int,
        default=90,
        help="Max characters per line (default: 90)",
    )
    parser.add_argument(
        "--lines-per-page",
        type=int,
        default=55,
        help="Max lines per page (default: 55)",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("data/_cuad"),
        help="Directory to cache CUAD data",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    cuad_json_path = _ensure_cuad_json(args.cache_dir)
    data = _load_json(cuad_json_path)

    docs = data.get("data", [])
    if not docs:
        raise ValueError("CUAD dataset is empty or invalid.")

    count = 0
    for idx, doc in enumerate(docs):
        if count >= args.limit:
            break
        title = doc.get("title", f"cuad_{idx}")
        paragraphs = doc.get("paragraphs", [])
        if not paragraphs:
            continue
        text = paragraphs[0].get("context", "")
        if not text.strip():
            continue
        safe_name = _safe_filename(title)
        pdf_path = args.out_dir / f"cuad_{idx:03d}_{safe_name}.pdf"
        _write_text_pdf(
            text=text,
            out_path=pdf_path,
            line_width=args.line_width,
            lines_per_page=args.lines_per_page,
        )
        count += 1

    print(f"Generated {count} PDFs in {args.out_dir}")


def _ensure_cuad_json(cache_dir: Path) -> Path:
    zip_path = cache_dir / "data.zip"
    json_path = cache_dir / "CUADv1.json"
    if not json_path.exists():
        if not zip_path.exists():
            print(f"Downloading CUAD data to {zip_path}")
            urllib.request.urlretrieve(CUAD_ZIP_URL, zip_path)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extract("CUADv1.json", cache_dir)
    return json_path


def _write_text_pdf(
    *,
    text: str,
    out_path: Path,
    line_width: int,
    lines_per_page: int,
) -> None:
    sanitized = _sanitize_text(text)
    wrapped = []
    for paragraph in sanitized.splitlines():
        paragraph = paragraph.strip()
        if not paragraph:
            wrapped.append("")
            continue
        wrapped.extend(textwrap.wrap(paragraph, width=line_width))

    pages = _split_lines(wrapped, lines_per_page)
    pdf_bytes = _build_pdf_bytes(pages)
    out_path.write_bytes(pdf_bytes)


def _build_pdf_bytes(pages: list[list[str]]) -> bytes:
    objects_by_id: dict[int, bytes] = {}

    page_ids = []
    next_id = 3
    font_id = next_id + (2 * len(pages))
    for page_lines in pages:
        content = _build_content_stream(page_lines)
        content_id = next_id
        page_id = next_id + 1
        next_id += 2

        content_obj = _pdf_obj(
            content_id,
            b"<< /Length %d >>\nstream\n%s\nendstream" % (len(content), content),
        )
        page_obj = _pdf_obj(
            page_id,
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            b"/Resources << /Font << /F1 %d 0 R >> >> /Contents %d 0 R >>"
            % (font_id, content_id),
        )
        objects_by_id[content_id] = content_obj
        objects_by_id[page_id] = page_obj
        page_ids.append(page_id)

    font_obj = _pdf_obj(font_id, b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    objects_by_id[font_id] = font_obj

    kids = " ".join(f"{pid} 0 R" for pid in page_ids).encode("ascii")
    pages_obj = _pdf_obj(
        2,
        b"<< /Type /Pages /Kids [ "
        + kids
        + b" ] /Count %d >>" % len(page_ids),
    )
    catalog_obj = _pdf_obj(1, b"<< /Type /Catalog /Pages 2 0 R >>")

    objects_by_id[1] = catalog_obj
    objects_by_id[2] = pages_obj

    objects = [objects_by_id[obj_id] for obj_id in sorted(objects_by_id.keys())]

    return _assemble_pdf(objects)


def _build_content_stream(lines: list[str]) -> bytes:
    out = ["BT", "/F1 10 Tf", "72 720 Td"]
    for line in lines:
        escaped = _pdf_escape(line)
        out.append(f"({escaped}) Tj")
        out.append("0 -12 Td")
    out.append("ET")
    return "\n".join(out).encode("ascii")


def _assemble_pdf(objects: list[bytes]) -> bytes:
    header = b"%PDF-1.4\n"
    body = bytearray()
    offsets = [0]

    for obj in objects:
        offsets.append(len(header) + len(body))
        body.extend(obj)
        body.extend(b"\n")

    xref_offset = len(header) + len(body)
    xref = bytearray()
    xref.extend(f"xref\n0 {len(objects)+1}\n".encode("ascii"))
    xref.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        xref.extend(f"{offset:010d} 00000 n \n".encode("ascii"))

    trailer = (
        f"trailer << /Size {len(objects)+1} /Root 1 0 R >>\n"
        f"startxref\n{xref_offset}\n%%EOF\n"
    ).encode("ascii")

    return header + body + xref + trailer


def _pdf_obj(obj_id: int, content: bytes) -> bytes:
    return f"{obj_id} 0 obj\n".encode("ascii") + content + b"\nendobj"


def _pdf_escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _split_lines(lines: list[str], lines_per_page: int) -> list[list[str]]:
    pages = []
    current = []
    for line in lines:
        current.append(line)
        if len(current) >= lines_per_page:
            pages.append(current)
            current = []
    if current:
        pages.append(current)
    return pages


def _sanitize_text(text: str) -> str:
    return "".join(ch if 32 <= ord(ch) <= 126 or ch in "\n\r\t" else " " for ch in text)


def _safe_filename(text: str) -> str:
    cleaned = "".join(ch if ch.isalnum() else "_" for ch in text.lower())
    cleaned = "_".join(part for part in cleaned.split("_") if part)
    return cleaned[:40] or "document"


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    main()
