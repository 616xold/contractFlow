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
    extract_fields_orchestrated,
    extract_fields_retrieval,
)

_LOW_COST_ORCHESTRATED_PROFILE = {
    "top_k": 2,
    "max_chunk_chars": 800,
    "chunk_max_chars": 1300,
    "max_repairs": 2,
    "disable_verifier": True,
    "risk_review_top_k": 2,
}


def main() -> None:
    repo_root = REPO_ROOT

    from dotenv import load_dotenv

    load_dotenv(repo_root / ".env")

    parser = argparse.ArgumentParser(description="ContractFlow extractor CLI (naive, retrieval, field agents, orchestrated).")
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
        "--field-agent-concurrency",
        type=int,
        default=4,
        help="Parallel workers for field-agent calls in field_agents/orchestrated modes (default: 4)",
    )
    parser.add_argument(
        "--repair-confidence-threshold",
        type=float,
        default=0.64,
        help="Orchestrated repair threshold (default: 0.64)",
    )
    parser.add_argument(
        "--max-repairs",
        type=int,
        default=3,
        help="Max orchestrated repair-agent passes (default: 3)",
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
        default=3,
        help="Max verifier-triggered repairs (default: 3)",
    )
    parser.add_argument(
        "--verifier-skip-confidence",
        type=float,
        default=0.82,
        help="Skip verifier for high-confidence fields above this threshold (default: 0.82)",
    )
    parser.add_argument(
        "--verifier-model",
        type=str,
        default=None,
        help="Optional model override for verifier/judge pass (default: same as --model)",
    )
    parser.add_argument(
        "--disable-cost-aware",
        action="store_true",
        help="Disable cost-aware early-exit heuristics in orchestrated mode.",
    )
    parser.add_argument(
        "--cost-aware-min-avg-confidence",
        type=float,
        default=0.82,
        help="Min average field confidence for cost-aware early-exit (default: 0.82).",
    )
    parser.add_argument(
        "--cost-aware-min-evidence-ratio",
        type=float,
        default=0.7,
        help="Min evidence ratio for cost-aware early-exit (default: 0.7).",
    )
    parser.add_argument(
        "--cost-aware-min-non-null-ratio",
        type=float,
        default=0.9,
        help="Min non-null ratio across non-nullable fields for cost-aware early-exit (default: 0.9).",
    )
    parser.add_argument(
        "--cost-aware-min-critical-confidence",
        type=float,
        default=0.75,
        help="Min confidence required across critical fields for cost-aware early-exit (default: 0.75).",
    )
    parser.add_argument(
        "--cost-aware-max-disagreement-rate",
        type=float,
        default=0.2,
        help="Max baseline disagreement rate tolerated for cost-aware early-exit (default: 0.2).",
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
        "--orchestrated-profile",
        type=str,
        choices=("default", "low_cost"),
        default="default",
        help="Optional orchestrated preset. low_cost applies tuned token-saving defaults.",
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
        orchestrated_overrides = _resolve_orchestrated_profile_overrides(args)
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
                reranker_model=args.reranker_model,
                reranker_top_n=args.reranker_top_n,
                top_k=args.top_k,
                max_chunk_chars=args.max_chunk_chars,
                chunk_max_chars=args.chunk_max_chars,
                field_agent_concurrency=args.field_agent_concurrency,
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
                reranker_model=args.reranker_model,
                reranker_top_n=args.reranker_top_n,
                top_k=orchestrated_overrides["top_k"],
                max_chunk_chars=orchestrated_overrides["max_chunk_chars"],
                chunk_max_chars=orchestrated_overrides["chunk_max_chars"],
                field_agent_concurrency=args.field_agent_concurrency,
                use_ocr=args.use_ocr,
                ocr_min_chars=args.ocr_min_chars,
                ocr_lang=args.ocr_lang,
                ocr_dpi=args.ocr_dpi,
                repair_confidence_threshold=args.repair_confidence_threshold,
                max_repairs=orchestrated_overrides["max_repairs"],
                enable_verifier=not orchestrated_overrides["disable_verifier"],
                verifier_confidence_threshold=args.verifier_confidence_threshold,
                verifier_max_repairs=args.verifier_max_repairs,
                verifier_skip_confidence=args.verifier_skip_confidence,
                verifier_model=args.verifier_model,
                enable_cost_aware=not args.disable_cost_aware,
                cost_aware_min_avg_confidence=args.cost_aware_min_avg_confidence,
                cost_aware_min_evidence_ratio=args.cost_aware_min_evidence_ratio,
                cost_aware_min_non_null_ratio=args.cost_aware_min_non_null_ratio,
                cost_aware_min_critical_confidence=args.cost_aware_min_critical_confidence,
                cost_aware_max_disagreement_rate=args.cost_aware_max_disagreement_rate,
                enable_risk_judge=not args.disable_risk_judge,
                enable_risk_review=not args.disable_risk_review,
                risk_judge_model=args.risk_judge_model,
                risk_review_model=args.risk_review_model,
                risk_review_top_k=orchestrated_overrides["risk_review_top_k"],
                risk_policy_path=args.risk_policy_path,
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
                enable_risk_judge=not args.disable_risk_judge,
                enable_risk_review=not args.disable_risk_review,
                risk_judge_model=args.risk_judge_model,
                risk_review_model=args.risk_review_model,
                risk_review_top_k=args.risk_review_top_k,
                risk_policy_path=args.risk_policy_path,
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


def _resolve_orchestrated_profile_overrides(args: argparse.Namespace) -> dict:
    params = {
        "top_k": int(args.top_k),
        "max_chunk_chars": int(args.max_chunk_chars),
        "chunk_max_chars": int(args.chunk_max_chars),
        "max_repairs": int(args.max_repairs),
        "disable_verifier": bool(args.disable_verifier),
        "risk_review_top_k": args.risk_review_top_k,
    }
    if args.orchestrated_profile == "low_cost":
        params.update(_LOW_COST_ORCHESTRATED_PROFILE)
    return params


if __name__ == "__main__":
    main()
