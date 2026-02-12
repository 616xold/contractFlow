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
    extract_fields_orchestrated,
    extract_fields_retrieval,
)


def main() -> None:
    repo_root = REPO_ROOT

    from dotenv import load_dotenv

    load_dotenv(repo_root / ".env")

    parser = argparse.ArgumentParser(description="Run ContractFlow extraction over all PDFs in a folder.")
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
    retrieval_group.add_argument(
        "--orchestrated",
        action="store_true",
        help="Use orchestrated extraction (global baseline + field agents + repair loop)",
    )
    parser.add_argument(
        "--retrieval-backend",
        type=str,
        default="bm25",
        help="Retrieval backend: bm25, embeddings, or hybrid (default: bm25)",
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
        "--reranker-model",
        type=str,
        default=None,
        help="Optional cross-encoder reranker model (requires sentence-transformers)",
    )
    parser.add_argument(
        "--reranker-top-n",
        type=int,
        default=20,
        help="Candidate pool size before reranking (default: 20)",
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
        "--disable-verifier",
        action="store_true",
        help="Disable verifier/judge pass in orchestrated mode",
    )
    parser.add_argument(
        "--verifier-confidence-threshold",
        type=float,
        default=0.62,
        help="Verifier confidence threshold before forcing revise (default: 0.62)",
    )
    parser.add_argument(
        "--verifier-max-repairs",
        type=int,
        default=4,
        help="Max verifier-triggered repairs (default: 4)",
    )
    parser.add_argument(
        "--verifier-model",
        type=str,
        default=None,
        help="Optional model override for verifier/judge pass (default: same as --model)",
    )
    parser.add_argument(
        "--disable-risk-judge",
        action="store_true",
        help="Disable risk judge arbitration (use rules-only risk scoring)",
    )
    parser.add_argument(
        "--risk-judge-model",
        type=str,
        default=None,
        help="Optional model override for risk judge (default: same as --model)",
    )
    parser.add_argument(
        "--disable-risk-review",
        action="store_true",
        help="Disable post-extraction risk review agent (rules + optional judge only)",
    )
    parser.add_argument(
        "--risk-review-model",
        type=str,
        default=None,
        help="Optional model override for risk review agent (default: same as --model)",
    )
    parser.add_argument(
        "--risk-review-top-k",
        type=int,
        default=None,
        help="Optional top-k override for risk review retrieval (default: from risk policy)",
    )
    parser.add_argument(
        "--risk-policy-path",
        type=Path,
        default=repo_root / "docs" / "risk_policy.json",
        help="Path to risk policy JSON (default: docs/risk_policy.json)",
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
                        embedding_cache_dir=args.embedding_cache_dir,
                        reranker_model=args.reranker_model,
                        reranker_top_n=args.reranker_top_n,
                        top_k=args.top_k,
                        max_chunk_chars=args.max_chunk_chars,
                        chunk_max_chars=args.chunk_max_chars,
                        use_ocr=args.use_ocr,
                        ocr_min_chars=args.ocr_min_chars,
                        ocr_lang=args.ocr_lang,
                        ocr_dpi=args.ocr_dpi,
                        enable_risk_judge=not args.disable_risk_judge,
                        enable_risk_review=not args.disable_risk_review,
                        risk_judge_model=args.risk_judge_model,
                        risk_review_model=args.risk_review_model,
                        risk_review_top_k=args.risk_review_top_k,
                        risk_policy_path=args.risk_policy_path,
                    )
                elif args.orchestrated:
                    result = extract_fields_orchestrated(
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
                        reranker_model=args.reranker_model,
                        reranker_top_n=args.reranker_top_n,
                        top_k=args.top_k,
                        max_chunk_chars=args.max_chunk_chars,
                        chunk_max_chars=args.chunk_max_chars,
                        use_ocr=args.use_ocr,
                        ocr_min_chars=args.ocr_min_chars,
                        ocr_lang=args.ocr_lang,
                        ocr_dpi=args.ocr_dpi,
                        enable_verifier=not args.disable_verifier,
                        verifier_confidence_threshold=args.verifier_confidence_threshold,
                        verifier_max_repairs=args.verifier_max_repairs,
                        verifier_model=args.verifier_model,
                        enable_risk_judge=not args.disable_risk_judge,
                        enable_risk_review=not args.disable_risk_review,
                        risk_judge_model=args.risk_judge_model,
                        risk_review_model=args.risk_review_model,
                        risk_review_top_k=args.risk_review_top_k,
                        risk_policy_path=args.risk_policy_path,
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
                        embedding_cache_dir=args.embedding_cache_dir,
                        reranker_model=args.reranker_model,
                        reranker_top_n=args.reranker_top_n,
                        top_k=args.top_k,
                        max_chunk_chars=args.max_chunk_chars,
                        chunk_max_chars=args.chunk_max_chars,
                        use_ocr=args.use_ocr,
                        ocr_min_chars=args.ocr_min_chars,
                        ocr_lang=args.ocr_lang,
                        ocr_dpi=args.ocr_dpi,
                        enable_risk_judge=not args.disable_risk_judge,
                        enable_risk_review=not args.disable_risk_review,
                        risk_judge_model=args.risk_judge_model,
                        risk_review_model=args.risk_review_model,
                        risk_review_top_k=args.risk_review_top_k,
                        risk_policy_path=args.risk_policy_path,
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
                        use_ocr=args.use_ocr,
                        ocr_min_chars=args.ocr_min_chars,
                        ocr_lang=args.ocr_lang,
                        ocr_dpi=args.ocr_dpi,
                        enable_risk_judge=not args.disable_risk_judge,
                        enable_risk_review=not args.disable_risk_review,
                        risk_judge_model=args.risk_judge_model,
                        risk_review_model=args.risk_review_model,
                        risk_review_top_k=args.risk_review_top_k,
                        risk_policy_path=args.risk_policy_path,
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
