"""Run extraction ablations and evaluate across modes."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
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
        default="naive,retrieval,field_agents,orchestrated",
        help="Comma-separated modes (naive,retrieval,field_agents,orchestrated)",
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
        help="Retrieval backend: bm25, embeddings, or hybrid (default: bm25)",
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
        "--risk-policy-path",
        type=Path,
        default=REPO_ROOT / "docs" / "risk_policy.json",
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
        "--bootstrap-samples",
        type=int,
        default=0,
        help="Bootstrap samples for 95 percent CI in evaluation summaries (default: 0)",
    )
    parser.add_argument(
        "--bootstrap-seed",
        type=int,
        default=42,
        help="Seed for bootstrap sampling (default: 42)",
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
    parser.add_argument(
        "--fixed-benchmark",
        action="store_true",
        help="Run the canonical 25-doc committee benchmark and emit a single report artifact.",
    )
    parser.add_argument(
        "--fixed-benchmark-out",
        type=Path,
        default=REPO_ROOT / "data" / "benchmarks" / "portfolio_benchmark.json",
        help="Output artifact path for --fixed-benchmark mode",
    )
    args = parser.parse_args()

    from dotenv import load_dotenv

    load_dotenv(REPO_ROOT / ".env")

    if args.fixed_benchmark:
        args.labels_dir = REPO_ROOT / "data" / "labels"
        args.label_suffix = ".silver_committee.json"
        args.modes = "naive,retrieval,field_agents,orchestrated"
        args.max_pdfs = 25
        args.bootstrap_samples = max(args.bootstrap_samples, 1000)
        if args.out is None:
            args.out = args.fixed_benchmark_out

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
            bootstrap_samples=args.bootstrap_samples,
            bootstrap_seed=args.bootstrap_seed,
        )
        results[mode] = summary

    pairwise = _build_pairwise_report(results, modes)
    _print_ablation_summary(results, pairwise)

    if args.out:
        output_payload = {
            "generated_at": datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
            "config": {
                "labels_dir": str(args.labels_dir),
                "label_suffix": args.label_suffix,
                "schema": str(args.schema),
                "modes": modes,
                "max_pdfs": args.max_pdfs,
                "skip_extraction": args.skip_extraction,
                "bootstrap_samples": args.bootstrap_samples,
                "bootstrap_seed": args.bootstrap_seed,
                "fixed_benchmark": args.fixed_benchmark,
                "disable_risk_judge": args.disable_risk_judge,
                "risk_judge_model": args.risk_judge_model,
                "risk_policy_path": str(args.risk_policy_path) if args.risk_policy_path else None,
            },
            "summaries": results,
            "pairwise_comparisons": pairwise,
        }
        for mode, summary in results.items():
            output_payload[mode] = summary
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(output_payload, indent=2) + "\n", encoding="utf-8")


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
                enable_risk_judge=not args.disable_risk_judge,
                risk_judge_model=args.risk_judge_model,
                risk_policy_path=args.risk_policy_path,
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
                risk_judge_model=args.risk_judge_model,
                risk_policy_path=args.risk_policy_path,
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
                risk_judge_model=args.risk_judge_model,
                risk_policy_path=args.risk_policy_path,
            )
        elif mode == "orchestrated":
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
                enable_risk_judge=not args.disable_risk_judge,
                risk_judge_model=args.risk_judge_model,
                risk_policy_path=args.risk_policy_path,
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


def _print_ablation_summary(results: dict, pairwise: dict) -> None:
    print("ablation_summary:")
    for mode, summary in results.items():
        print(
            f"  {mode}: exact={summary['overall_accuracy_exact']} "
            f"partial={summary['overall_accuracy_partial']} "
            f"docs={summary['docs_evaluated']}"
        )
    if pairwise:
        print("pairwise_comparisons:")
        for pair_key, stats in pairwise.items():
            exact = stats["exact"]
            partial = stats["partial"]
            print(
                f"  {pair_key}: exact_delta={exact['mean_delta']} "
                f"(w/t/l={exact['wins']}/{exact['ties']}/{exact['losses']}), "
                f"partial_delta={partial['mean_delta']} "
                f"(w/t/l={partial['wins']}/{partial['ties']}/{partial['losses']})"
            )


def _build_pairwise_report(results: dict, modes: list[str]) -> dict:
    pairwise: dict = {}
    for idx, baseline in enumerate(modes):
        for challenger in modes[idx + 1 :]:
            base_summary = results.get(baseline, {})
            chall_summary = results.get(challenger, {})
            pair_key = f"{baseline}__vs__{challenger}"
            pairwise[pair_key] = _compare_mode_pair(base_summary, chall_summary, baseline, challenger)
    return pairwise


def _compare_mode_pair(
    baseline_summary: dict,
    challenger_summary: dict,
    baseline_name: str,
    challenger_name: str,
) -> dict:
    baseline_docs = _index_doc_metrics(baseline_summary.get("docs", []))
    challenger_docs = _index_doc_metrics(challenger_summary.get("docs", []))
    common_docs = sorted(set(baseline_docs.keys()) & set(challenger_docs.keys()))

    doc_deltas = []
    deltas_exact: list[float] = []
    deltas_partial: list[float] = []
    wins_exact = ties_exact = losses_exact = 0
    wins_partial = ties_partial = losses_partial = 0

    for doc in common_docs:
        base_doc = baseline_docs[doc]
        chall_doc = challenger_docs[doc]
        delta_exact = float(chall_doc["accuracy_exact"]) - float(base_doc["accuracy_exact"])
        delta_partial = float(chall_doc["accuracy_partial"]) - float(base_doc["accuracy_partial"])
        deltas_exact.append(delta_exact)
        deltas_partial.append(delta_partial)

        outcome_exact = _delta_outcome(delta_exact)
        outcome_partial = _delta_outcome(delta_partial)
        if outcome_exact == "win":
            wins_exact += 1
        elif outcome_exact == "loss":
            losses_exact += 1
        else:
            ties_exact += 1

        if outcome_partial == "win":
            wins_partial += 1
        elif outcome_partial == "loss":
            losses_partial += 1
        else:
            ties_partial += 1

        doc_deltas.append(
            {
                "doc": doc,
                "baseline_accuracy_exact": round(float(base_doc["accuracy_exact"]), 4),
                "challenger_accuracy_exact": round(float(chall_doc["accuracy_exact"]), 4),
                "delta_exact": round(delta_exact, 4),
                "outcome_exact": outcome_exact,
                "baseline_accuracy_partial": round(float(base_doc["accuracy_partial"]), 4),
                "challenger_accuracy_partial": round(float(chall_doc["accuracy_partial"]), 4),
                "delta_partial": round(delta_partial, 4),
                "outcome_partial": outcome_partial,
            }
        )

    field_delta_exact = _field_accuracy_delta(
        baseline_summary.get("field_accuracy", {}),
        challenger_summary.get("field_accuracy", {}),
        metric="accuracy_exact",
    )
    field_delta_partial = _field_accuracy_delta(
        baseline_summary.get("field_accuracy", {}),
        challenger_summary.get("field_accuracy", {}),
        metric="accuracy_partial",
    )

    return {
        "baseline": baseline_name,
        "challenger": challenger_name,
        "docs_compared": len(common_docs),
        "exact": {
            "mean_delta": round(_mean(deltas_exact), 4) if deltas_exact else 0.0,
            "wins": wins_exact,
            "ties": ties_exact,
            "losses": losses_exact,
        },
        "partial": {
            "mean_delta": round(_mean(deltas_partial), 4) if deltas_partial else 0.0,
            "wins": wins_partial,
            "ties": ties_partial,
            "losses": losses_partial,
        },
        "field_delta_exact": field_delta_exact,
        "field_delta_partial": field_delta_partial,
        "doc_deltas": doc_deltas,
    }


def _index_doc_metrics(docs: list[dict]) -> dict:
    out: dict = {}
    for doc in docs:
        if doc.get("status") != "evaluated":
            continue
        name = doc.get("doc")
        if not isinstance(name, str) or not name:
            continue
        out[name] = doc
    return out


def _field_accuracy_delta(baseline_field: dict, challenger_field: dict, *, metric: str) -> dict:
    out: dict = {}
    for field in sorted(set(baseline_field.keys()) | set(challenger_field.keys())):
        base = float((baseline_field.get(field) or {}).get(metric, 0.0))
        chall = float((challenger_field.get(field) or {}).get(metric, 0.0))
        out[field] = round(chall - base, 4)
    return out


def _delta_outcome(delta: float, eps: float = 1e-9) -> str:
    if delta > eps:
        return "win"
    if delta < -eps:
        return "loss"
    return "tie"


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


if __name__ == "__main__":
    main()
