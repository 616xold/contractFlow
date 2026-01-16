"""Bulk runner for baseline extraction over a directory of PDFs."""

from __future__ import annotations

import argparse
import csv
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

    parser = argparse.ArgumentParser(description="Run baseline extraction over all PDFs in a folder.")
    parser.add_argument(
        "--in-dir",
        type=Path,
        default=repo_root / "data" / "raw_pdfs",
        help="Directory containing input PDFs",
    )
    parser.add_argument(
        "--preds-dir",
        type=Path,
        default=repo_root / "data" / "preds",
        help="Directory to write prediction outputs",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Path to write a CSV summary (default: <preds-dir>/summary.csv)",
    )
    parser.add_argument(
        "--schema",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "contractflow" / "schemas" / "contract_schema.json",
        help="Path to the JSON schema describing fields to extract",
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
    args = parser.parse_args()

    pdf_paths = sorted(args.in_dir.glob("*.pdf"))
    if not pdf_paths:
        print(f"No PDFs found in {args.in_dir}", file=sys.stderr)
        raise SystemExit(1)

    args.preds_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.csv or (args.preds_dir / "summary.csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "pdf",
        "model",
        "input_tokens",
        "output_tokens",
        "success",
        "error",
    ]

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for pdf_path in pdf_paths:
            out_path = args.preds_dir / f"{pdf_path.stem}.pred.json"
            raw_out_path = args.preds_dir / f"{pdf_path.stem}.raw.txt"

            try:
                result: ExtractionResult
                if args.field_agents:
                    result = extract_fields_field_agents(
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
                        top_k=args.top_k,
                        max_chunk_chars=args.max_chunk_chars,
                    )
                elif args.retrieval:
                    result = extract_fields_retrieval(
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
                        top_k=args.top_k,
                        max_chunk_chars=args.max_chunk_chars,
                    )
                else:
                    result = extract_fields_naive(
                        pdf_path,
                        args.schema,
                        model=args.model,
                        validate=not args.no_validate,
                        strict=args.strict,
                        coerce=not args.no_coerce,
                        structured_outputs=not args.no_structured_outputs,
                    )
                raw_out_path.write_text(result.raw_text, encoding="utf-8")

                pred_payload: dict = dict(result.json_result)
                pred_payload["_meta"] = {
                    "pdf": str(pdf_path),
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
                out_path.write_text(
                    json.dumps(pred_payload, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8",
                )
                writer.writerow(
                    {
                        "pdf": pdf_path.name,
                        "model": args.model,
                        "input_tokens": result.prompt_tokens,
                        "output_tokens": result.completion_tokens,
                        "success": 1,
                        "error": "",
                    }
                )
                if result.issues:
                    print(f"WARN {pdf_path.name} issues={len(result.issues)}", file=sys.stderr)
                else:
                    print(f"OK   {pdf_path.name}", file=sys.stderr)
            except Exception as e:
                error = str(e).replace("\r\n", "\\n").replace("\n", "\\n")
                writer.writerow(
                    {
                        "pdf": pdf_path.name,
                        "model": args.model,
                        "input_tokens": "",
                        "output_tokens": "",
                        "success": 0,
                        "error": error,
                    }
                )
                print(f"FAIL {pdf_path.name}: {e}", file=sys.stderr)

    print(f"Wrote summary CSV to {csv_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
