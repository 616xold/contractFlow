"""Bootstrap label files using an extraction mode (silver labels)."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

# Ensure repo root is on PYTHONPATH when running as a script.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from contractflow.core.extractor import (
    DEFAULT_MODEL,
    ExtractionResult,
    extract_fields_field_agents,
    extract_fields_naive,
    extract_fields_retrieval,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Bootstrap label files using an extraction mode.")
    parser.add_argument(
        "--in-dir",
        type=Path,
        default=REPO_ROOT / "data" / "raw_pdfs",
        help="Directory containing input PDFs",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "data" / "labels",
        help="Directory to write labels",
    )
    parser.add_argument(
        "--schema",
        type=Path,
        default=REPO_ROOT / "contractflow" / "schemas" / "contract_schema.json",
        help="Path to the JSON schema describing fields to extract",
    )
    parser.add_argument(
        "--label-suffix",
        type=str,
        default=".silver.json",
        help="Label filename suffix (default: .silver.json)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="field_agents",
        choices=["naive", "retrieval", "field_agents"],
        help="Extraction mode to use for labeling",
    )
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="OpenAI model name")
    parser.add_argument("--no-validate", action="store_true", help="Disable schema validation")
    parser.add_argument("--strict", action="store_true", help="Fail fast on schema issues")
    parser.add_argument("--no-coerce", action="store_true", help="Disable type coercion")
    parser.add_argument(
        "--no-structured-outputs",
        action="store_true",
        help="Disable structured outputs parsing",
    )
    parser.add_argument(
        "--retrieval-backend",
        type=str,
        default="bm25",
        help="Retrieval backend (default: bm25)",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="text-embedding-3-small",
        help="Embedding model for embeddings backend",
    )
    parser.add_argument(
        "--embedding-batch-size",
        type=int,
        default=64,
        help="Embedding batch size for indexing",
    )
    parser.add_argument(
        "--embedding-cache-dir",
        type=Path,
        default=REPO_ROOT / "data" / "embeddings",
        help="Directory for embedding cache files",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of chunks to retrieve per field",
    )
    parser.add_argument(
        "--max-chunk-chars",
        type=int,
        default=1200,
        help="Max chars per chunk in the prompt",
    )
    parser.add_argument(
        "--chunk-max-chars",
        type=int,
        default=2000,
        help="Max chars per chunk during chunking",
    )
    parser.add_argument(
        "--use-ocr",
        action="store_true",
        help="Enable OCR fallback when extracted text is sparse",
    )
    parser.add_argument(
        "--ocr-min-chars",
        type=int,
        default=40,
        help="Min avg chars per page before OCR fallback",
    )
    parser.add_argument(
        "--ocr-lang",
        type=str,
        default="eng",
        help="OCR language (default: eng)",
    )
    parser.add_argument(
        "--ocr-dpi",
        type=int,
        default=200,
        help="OCR DPI for pdf2image (default: 200)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing labels",
    )
    parser.add_argument(
        "--max-pdfs",
        type=int,
        default=None,
        help="Optional limit on number of PDFs to process",
    )
    args = parser.parse_args()

    from dotenv import load_dotenv

    load_dotenv(REPO_ROOT / ".env")

    pdf_paths = sorted(args.in_dir.glob("*.pdf"))
    if args.max_pdfs is not None:
        pdf_paths = pdf_paths[: max(0, args.max_pdfs)]
    if not pdf_paths:
        print(f"No PDFs found in {args.in_dir}", file=sys.stderr)
        raise SystemExit(1)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out_dir / "manifest.json"
    manifest = _load_manifest(manifest_path)

    for pdf_path in pdf_paths:
        label_path = args.out_dir / f"{pdf_path.stem}{args.label_suffix}"
        if label_path.exists() and not args.overwrite:
            continue
        result = _extract_label(pdf_path, args)
        label_path.write_text(json.dumps(result.json_result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

        manifest_entry = {
            "doc": pdf_path.stem,
            "label_file": str(label_path),
            "label_suffix": args.label_suffix,
            "label_quality": "silver",
            "mode": args.mode,
            "model": args.model,
            "created_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        }
        manifest[pdf_path.stem] = manifest_entry

    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote labels to {args.out_dir} using mode={args.mode}")


def _extract_label(pdf_path: Path, args: argparse.Namespace) -> ExtractionResult:
    if args.mode == "naive":
        return extract_fields_naive(
            pdf_path,
            args.schema,
            model=args.model,
            validate=not args.no_validate,
            strict=args.strict,
            coerce=not args.no_coerce,
            structured_outputs=not args.no_structured_outputs,
            use_ocr=args.use_ocr,
            ocr_min_chars=args.ocr_min_chars,
            ocr_lang=args.ocr_lang,
            ocr_dpi=args.ocr_dpi,
        )
    if args.mode == "retrieval":
        return extract_fields_retrieval(
            pdf_path,
            args.schema,
            model=args.model,
            validate=not args.no_validate,
            strict=args.strict,
            coerce=not args.no_coerce,
            structured_outputs=not args.no_structured_outputs,
            retrieval_backend=args.retrieval_backend,
            embedding_model=args.embedding_model,
            embedding_batch_size=args.embedding_batch_size,
            embedding_cache_dir=args.embedding_cache_dir,
            top_k=args.top_k,
            max_chunk_chars=args.max_chunk_chars,
            chunk_max_chars=args.chunk_max_chars,
            use_ocr=args.use_ocr,
            ocr_min_chars=args.ocr_min_chars,
            ocr_lang=args.ocr_lang,
            ocr_dpi=args.ocr_dpi,
        )
    return extract_fields_field_agents(
        pdf_path,
        args.schema,
        model=args.model,
        validate=not args.no_validate,
        strict=args.strict,
        coerce=not args.no_coerce,
        structured_outputs=not args.no_structured_outputs,
        retrieval_backend=args.retrieval_backend,
        embedding_model=args.embedding_model,
        embedding_batch_size=args.embedding_batch_size,
        embedding_cache_dir=args.embedding_cache_dir,
        top_k=args.top_k,
        max_chunk_chars=args.max_chunk_chars,
        chunk_max_chars=args.chunk_max_chars,
        use_ocr=args.use_ocr,
        ocr_min_chars=args.ocr_min_chars,
        ocr_lang=args.ocr_lang,
        ocr_dpi=args.ocr_dpi,
    )


def _load_manifest(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


if __name__ == "__main__":
    main()
