"""CLI for baseline single-call extraction."""

from __future__ import annotations

import argparse
import json
import sys
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
    repo_root = REPO_ROOT

    from dotenv import load_dotenv

    load_dotenv(repo_root / ".env")

    parser = argparse.ArgumentParser(description="Baseline ContractFlow extractor (single LLM call).")
    parser.add_argument("pdf_path", type=Path, help="Path to the contract PDF")
    parser.add_argument(
        "--schema",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "contractflow" / "schemas" / "contract_schema.json",
        help="Path to the JSON schema describing fields to extract",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Write parsed prediction JSON to this path (default: data/preds/<pdf_stem>.pred.json)",
    )
    parser.add_argument(
        "--raw-out",
        type=Path,
        default=None,
        help="Write raw model output to this path (default: data/preds/<pdf_stem>.raw.txt)",
    )
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="OpenAI model name")
    parser.add_argument("--no-validate", action="store_true", help="Disable schema validation (debugging)")
    parser.add_argument("--strict", action="store_true", help="Fail fast on schema issues (default: lenient)")
    parser.add_argument("--no-coerce", action="store_true", help="Disable type coercion (debugging)")
    parser.add_argument(
        "--no-structured-outputs",
        action="store_true",
        help="Disable structured outputs parsing (debugging / fallback mode)",
    )
    retrieval_group = parser.add_mutually_exclusive_group()
    retrieval_group.add_argument(
        "--retrieval",
        action="store_true",
        help="Use retrieval context over chunked pages for a single LLM call",
    )
    retrieval_group.add_argument(
        "--field-agents",
        action="store_true",
        help="Use per-field retrieval and extraction agents",
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
        help="Embedding model for embeddings backend (default: text-embedding-3-small)",
    )
    parser.add_argument(
        "--embedding-batch-size",
        type=int,
        default=64,
        help="Embedding batch size for indexing (default: 64)",
    )
    parser.add_argument(
        "--embedding-cache-dir",
        type=Path,
        default=repo_root / "data" / "embeddings",
        help="Directory for embedding cache files (default: data/embeddings)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of chunks to retrieve per field (default: 3)",
    )
    parser.add_argument(
        "--max-chunk-chars",
        type=int,
        default=1200,
        help="Max chars per chunk in the prompt (default: 1200)",
    )
    parser.add_argument(
        "--chunk-max-chars",
        type=int,
        default=2000,
        help="Max chars per chunk during chunking (default: 2000)",
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
        help="Min avg chars per page before OCR fallback (default: 40)",
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
    args = parser.parse_args()

    preds_dir = repo_root / "data" / "preds"
    out_path = args.out or (preds_dir / f"{args.pdf_path.stem}.pred.json")
    raw_out_path = args.raw_out or (preds_dir / f"{args.pdf_path.stem}.raw.txt")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    raw_out_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        result: ExtractionResult
        if args.field_agents:
            result = extract_fields_field_agents(
                args.pdf_path,
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
        elif args.retrieval:
            result = extract_fields_retrieval(
                args.pdf_path,
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
        else:
            result = extract_fields_naive(
                args.pdf_path,
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
    except Exception as e:
        print(f"Extraction failed: {e}", file=sys.stderr)
        raise SystemExit(1) from e

    raw_out_path.write_text(result.raw_text, encoding="utf-8")

    pred_payload: dict = dict(result.json_result)
    pred_payload["_meta"] = {
        "pdf": str(args.pdf_path),
        "model": args.model,
        "input_tokens": result.prompt_tokens,
        "output_tokens": result.completion_tokens,
        "validate": not args.no_validate,
        "strict": args.strict,
        "coerce": not args.no_coerce,
        "structured_outputs": not args.no_structured_outputs,
        "issues": result.issues or [],
    }
    pred_payload["_meta"]["retrieval"] = result.retrieval or {"enabled": False}
    out_path.write_text(json.dumps(pred_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(
        f"model={args.model} input_tokens={result.prompt_tokens} output_tokens={result.completion_tokens}",
        file=sys.stderr,
    )
    if result.issues:
        print(f"validation_issues={len(result.issues)}", file=sys.stderr)
    print(f"wrote_pred={out_path}", file=sys.stderr)
    print(f"wrote_raw={raw_out_path}", file=sys.stderr)

    print(json.dumps(result.json_result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
