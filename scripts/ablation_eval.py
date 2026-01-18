"""Run extraction ablations and evaluate across modes."""

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
from scripts.evaluate import evaluate_predictions


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ablations and evaluate accuracy.")
    parser.add_argument(
        "--in-dir",
        type=Path,
        default=REPO_ROOT / "data" / "raw_pdfs",
        help="Directory containing input PDFs",
    )
    parser.add_argument(
        "--labels-dir",
        type=Path,
        default=REPO_ROOT / "data" / "labels",
        help="Directory containing label files",
    )
    parser.add_argument(
        "--label-suffix",
        type=str,
        default=".gold.json",
        help="Label filename suffix (default: .gold.json)",
    )
    parser.add_argument(
        "--schema",
        type=Path,
        default=REPO_ROOT / "contractflow" / "schemas" / "contract_schema.json",
        help="Path to the JSON schema describing fields to extract",
    )
    parser.add_argument(
        "--preds-root",
        type=Path,
        default=REPO_ROOT / "data" / "preds_ablations",
        help="Root folder for ablation predictions",
    )
    parser.add_argument(
        "--modes",
        type=str,
        default="naive,retrieval,field_agents",
        help="Comma-separated modes (naive,retrieval,field_agents)",
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
        "--partial-threshold",
        type=float,
        default=0.85,
        help="Partial match threshold for evaluation",
    )
    parser.add_argument(
        "--skip-extraction",
        action="store_true",
        help="Skip extraction and only run evaluation",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing predictions during extraction",
    )
    parser.add_argument(
        "--max-pdfs",
        type=int,
        default=None,
        help="Optional limit on number of PDFs to process",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional path to write JSON summary",
    )
    args = parser.parse_args()

    from dotenv import load_dotenv

    load_dotenv(REPO_ROOT / ".env")

    modes = [mode.strip() for mode in args.modes.split(",") if mode.strip()]
    pdf_paths = sorted(args.in_dir.glob("*.pdf"))
    if args.max_pdfs is not None:
        pdf_paths = pdf_paths[: max(0, args.max_pdfs)]
    if not pdf_paths:
        print(f"No PDFs found in {args.in_dir}", file=sys.stderr)
        raise SystemExit(1)

    results = {}
    for mode in modes:
        preds_dir = args.preds_root / mode
        raws_dir = preds_dir / "raw"
        preds_dir.mkdir(parents=True, exist_ok=True)
        raws_dir.mkdir(parents=True, exist_ok=True)

        if not args.skip_extraction:
            _run_mode(
                mode,
                pdf_paths,
                preds_dir,
                raws_dir,
                args,
            )

        summary = evaluate_predictions(
            labels_dir=args.labels_dir,
            preds_dir=preds_dir,
            schema_path=args.schema,
            label_suffix=args.label_suffix,
            partial_threshold=args.partial_threshold,
        )
        results[mode] = summary

    _print_ablation_summary(results)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")


def _run_mode(
    mode: str,
    pdf_paths: list[Path],
    preds_dir: Path,
    raws_dir: Path,
    args: argparse.Namespace,
) -> None:
    for pdf_path in pdf_paths:
        out_path = preds_dir / f"{pdf_path.stem}.pred.json"
        raw_out_path = raws_dir / f"{pdf_path.stem}.raw.txt"
        if out_path.exists() and not args.overwrite:
            continue

        if mode == "naive":
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
            )
        elif mode == "retrieval":
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
                top_k=args.top_k,
                max_chunk_chars=args.max_chunk_chars,
                chunk_max_chars=args.chunk_max_chars,
                use_ocr=args.use_ocr,
                ocr_min_chars=args.ocr_min_chars,
                ocr_lang=args.ocr_lang,
                ocr_dpi=args.ocr_dpi,
            )
        elif mode == "field_agents":
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
                top_k=args.top_k,
                max_chunk_chars=args.max_chunk_chars,
                chunk_max_chars=args.chunk_max_chars,
                use_ocr=args.use_ocr,
                ocr_min_chars=args.ocr_min_chars,
                ocr_lang=args.ocr_lang,
                ocr_dpi=args.ocr_dpi,
            )
        else:
            raise ValueError(f"Unknown mode: {mode}")

        _write_outputs(out_path, raw_out_path, pdf_path, args, mode, result)


def _write_outputs(
    out_path: Path,
    raw_out_path: Path,
    pdf_path: Path,
    args: argparse.Namespace,
    mode: str,
    result: ExtractionResult,
) -> None:
    raw_out_path.write_text(result.raw_text or "", encoding="utf-8")

    pred_payload: dict = dict(result.json_result)
    pred_payload["_meta"] = {
        "pdf": str(pdf_path),
        "model": args.model,
        "mode": mode,
        "input_tokens": result.prompt_tokens,
        "output_tokens": result.completion_tokens,
        "validate": not args.no_validate,
        "strict": args.strict,
        "coerce": not args.no_coerce,
        "structured_outputs": not args.no_structured_outputs,
        "issues": result.issues or [],
        "retrieval": result.retrieval or {"enabled": False},
    }
    out_path.write_text(json.dumps(pred_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _print_ablation_summary(results: dict) -> None:
    print("ablation_summary:")
    for mode, summary in results.items():
        print(
            f"  {mode}: exact={summary['overall_accuracy_exact']} "
            f"partial={summary['overall_accuracy_partial']} "
            f"docs={summary['docs_evaluated']}"
        )


if __name__ == "__main__":
    main()
