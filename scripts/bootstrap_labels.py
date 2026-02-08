"""Bootstrap label files using an extraction mode (silver labels)."""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict, Optional

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
        default=".silver_committee.json",
        help="Label filename suffix (default: .silver_committee.json)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="committee",
        choices=["naive", "retrieval", "field_agents", "orchestrated", "committee"],
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
        "--overwrite",
        action="store_true",
        help="Overwrite existing labels",
    )
    parser.add_argument(
        "--preds-root",
        type=Path,
        default=REPO_ROOT / "data" / "preds_ablations",
        help="Root folder containing per-mode prediction folders for committee labels",
    )
    parser.add_argument(
        "--committee-modes",
        type=str,
        default="orchestrated,field_agents,retrieval,naive",
        help="Comma-separated mode order for committee labels",
    )
    parser.add_argument(
        "--committee-orch-min-confidence",
        type=float,
        default=0.68,
        help="Minimum orchestrated confidence for committee selection",
    )
    parser.add_argument(
        "--committee-field-min-confidence",
        type=float,
        default=0.66,
        help="Minimum field-agents confidence for committee selection",
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
    schema = json.loads(args.schema.read_text(encoding="utf-8"))

    for pdf_path in pdf_paths:
        label_path = args.out_dir / f"{pdf_path.stem}{args.label_suffix}"
        if label_path.exists() and not args.overwrite:
            continue
        committee_meta: Optional[Dict[str, Any]] = None
        if args.mode == "committee":
            label_payload, committee_meta = _build_committee_label(pdf_path.stem, schema, args)
        else:
            result = _extract_label(pdf_path, args)
            label_payload = result.json_result
        label_path.write_text(json.dumps(label_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

        manifest_entry = {
            "doc": pdf_path.stem,
            "label_file": str(label_path),
            "label_suffix": args.label_suffix,
            "label_quality": "silver",
            "mode": args.mode,
            "model": args.model,
            "created_at": datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
        }
        if committee_meta is not None:
            manifest_entry["teacher"] = committee_meta
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
            enable_risk_judge=not args.disable_risk_judge,
            risk_judge_model=args.risk_judge_model,
            risk_policy_path=args.risk_policy_path,
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
    if args.mode == "orchestrated":
        return extract_fields_orchestrated(
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


_MODE_PRIORITY = {
    "orchestrated": 0,
    "field_agents": 1,
    "retrieval": 2,
    "naive": 3,
}
_DEFAULT_MODE_CONFIDENCE = {
    "orchestrated": 0.70,
    "field_agents": 0.66,
    "retrieval": 0.60,
    "naive": 0.52,
}


def _build_committee_label(
    doc_stem: str,
    schema: Dict[str, Any],
    args: argparse.Namespace,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    requested_modes = _parse_mode_list(args.committee_modes)
    available_preds: Dict[str, Dict[str, Any]] = {}
    missing_modes: list[str] = []

    for mode in requested_modes:
        pred_path = args.preds_root / mode / f"{doc_stem}.pred.json"
        payload = _load_prediction_file(pred_path)
        if payload is None:
            missing_modes.append(mode)
            continue
        available_preds[mode] = payload

    if not available_preds:
        raise ValueError(
            f"No committee prediction files found for '{doc_stem}' under {args.preds_root}"
        )

    label_payload: Dict[str, Any] = {}
    reason_counts: Dict[str, int] = {}
    mode_counts: Dict[str, int] = {}

    for field, meta in schema.items():
        candidates: list[Dict[str, Any]] = []
        for mode in requested_modes:
            pred = available_preds.get(mode)
            if pred is None:
                continue
            raw_value = pred.get(field)
            normalized_value = _normalize_for_committee(raw_value, meta)
            confidence = _extract_mode_confidence(pred, field, mode)
            evidence_count = _extract_mode_evidence_count(pred, field)
            candidates.append(
                {
                    "mode": mode,
                    "raw": raw_value,
                    "normalized": normalized_value,
                    "confidence": confidence,
                    "evidence_count": evidence_count,
                    "priority": _MODE_PRIORITY.get(mode, 99),
                }
            )

        if not candidates:
            label_payload[field] = None
            reason_counts["no_candidates"] = reason_counts.get("no_candidates", 0) + 1
            continue

        selected = _select_committee_candidate(
            candidates,
            orch_min_confidence=args.committee_orch_min_confidence,
            field_min_confidence=args.committee_field_min_confidence,
        )
        selected_mode = str(selected["mode"])
        selected_reason = str(selected["reason"])
        selected_raw = selected["raw"]
        selected_norm = selected["normalized"]

        field_type = meta.get("type")
        if field_type in {"integer", "boolean"}:
            label_payload[field] = selected_norm
        else:
            label_payload[field] = selected_raw

        reason_counts[selected_reason] = reason_counts.get(selected_reason, 0) + 1
        mode_counts[selected_mode] = mode_counts.get(selected_mode, 0) + 1

    teacher_meta = {
        "type": "committee",
        "strategy": "consensus_then_orch_field_bias",
        "modes_requested": requested_modes,
        "modes_available": list(available_preds.keys()),
        "modes_missing": missing_modes,
        "mode_selection_counts": mode_counts,
        "reason_counts": reason_counts,
        "preds_root": str(args.preds_root),
    }
    return label_payload, teacher_meta


def _select_committee_candidate(
    candidates: list[Dict[str, Any]],
    *,
    orch_min_confidence: float,
    field_min_confidence: float,
) -> Dict[str, Any]:
    informative = [c for c in candidates if _is_informative_value(c["normalized"])]
    groups: Dict[Any, list[Dict[str, Any]]] = {}
    for candidate in informative:
        groups.setdefault(candidate["normalized"], []).append(candidate)

    consensus_groups = [group for group in groups.values() if len(group) >= 2]
    if consensus_groups:
        def _group_key(group: list[Dict[str, Any]]) -> tuple[int, int, float, int]:
            high_tier = sum(
                1 for item in group if item["mode"] in {"orchestrated", "field_agents"}
            )
            total_conf = sum(float(item["confidence"]) for item in group)
            best_priority = min(int(item["priority"]) for item in group)
            return (len(group), high_tier, total_conf, -best_priority)

        winning_group = max(consensus_groups, key=_group_key)
        winner = min(
            winning_group,
            key=lambda item: (
                int(item["priority"]),
                -float(item["confidence"]),
                -int(item["evidence_count"]),
            ),
        )
        winner = dict(winner)
        winner["reason"] = "multi_mode_consensus"
        return winner

    by_mode = {str(item["mode"]): item for item in candidates}
    orch = by_mode.get("orchestrated")
    field = by_mode.get("field_agents")
    retrieval = by_mode.get("retrieval")
    naive = by_mode.get("naive")

    if (
        orch is not None
        and field is not None
        and _is_informative_value(orch["normalized"])
        and _is_informative_value(field["normalized"])
        and orch["normalized"] == field["normalized"]
    ):
        winner = dict(orch)
        winner["reason"] = "orch_field_agree"
        return winner

    if (
        orch is not None
        and field is not None
        and _is_informative_value(orch["normalized"])
        and _is_informative_value(field["normalized"])
        and float(orch["confidence"]) >= orch_min_confidence
        and float(field["confidence"]) >= field_min_confidence
    ):
        winner = orch if float(orch["confidence"]) >= float(field["confidence"]) else field
        winner = dict(winner)
        winner["reason"] = "orch_field_high_conf_disagree"
        return winner

    if (
        orch is not None
        and _is_informative_value(orch["normalized"])
        and float(orch["confidence"]) >= orch_min_confidence
    ):
        winner = dict(orch)
        winner["reason"] = "orch_high_conf"
        return winner

    if (
        field is not None
        and _is_informative_value(field["normalized"])
        and float(field["confidence"]) >= field_min_confidence
    ):
        winner = dict(field)
        winner["reason"] = "field_high_conf"
        return winner

    for mode_name, reason in (
        ("retrieval", "retrieval_fallback"),
        ("orchestrated", "orch_fallback"),
        ("field_agents", "field_fallback"),
        ("naive", "naive_fallback"),
    ):
        item = by_mode.get(mode_name)
        if item is not None and _is_informative_value(item["normalized"]):
            winner = dict(item)
            winner["reason"] = reason
            return winner

    winner = dict(candidates[0])
    winner["reason"] = "raw_fallback"
    return winner


def _parse_mode_list(raw: str) -> list[str]:
    modes = [item.strip() for item in raw.split(",") if item.strip()]
    out: list[str] = []
    seen: set[str] = set()
    for mode in modes:
        if mode not in _MODE_PRIORITY:
            continue
        if mode in seen:
            continue
        out.append(mode)
        seen.add(mode)
    return out or ["orchestrated", "field_agents", "retrieval", "naive"]


def _load_prediction_file(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    if not isinstance(data, dict):
        return None
    return data


def _extract_mode_confidence(payload: Dict[str, Any], field: str, mode: str) -> float:
    meta = payload.get("_meta")
    if isinstance(meta, dict):
        retrieval_meta = meta.get("retrieval")
        if isinstance(retrieval_meta, dict):
            fields_meta = retrieval_meta.get("fields")
            if isinstance(fields_meta, dict):
                field_meta = fields_meta.get(field)
                if isinstance(field_meta, dict):
                    confidence = field_meta.get("confidence")
                    if isinstance(confidence, (int, float)):
                        return float(confidence)
    return _DEFAULT_MODE_CONFIDENCE.get(mode, 0.5)


def _extract_mode_evidence_count(payload: Dict[str, Any], field: str) -> int:
    meta = payload.get("_meta")
    if not isinstance(meta, dict):
        return 0
    retrieval_meta = meta.get("retrieval")
    if not isinstance(retrieval_meta, dict):
        return 0
    fields_meta = retrieval_meta.get("fields")
    if not isinstance(fields_meta, dict):
        return 0
    field_meta = fields_meta.get(field)
    if not isinstance(field_meta, dict):
        return 0
    evidence = field_meta.get("evidence")
    if not isinstance(evidence, list):
        return 0
    return len(evidence)


def _normalize_for_committee(value: Any, meta: Dict[str, Any]) -> Any:
    expected = meta.get("type")
    enum_vals = meta.get("enum")
    if value is None:
        return None

    if expected == "integer":
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, int):
            return value
        if isinstance(value, float) and value.is_integer():
            return int(value)
        if isinstance(value, str):
            return _first_int(value)
        return None

    if expected == "boolean":
        if isinstance(value, bool):
            return value
        if isinstance(value, int) and value in (0, 1):
            return bool(value)
        if isinstance(value, str):
            cleaned = value.strip().lower()
            if cleaned in {"true", "t", "yes", "y", "1"}:
                return True
            if cleaned in {"false", "f", "no", "n", "0"}:
                return False
        return None

    text = " ".join(str(value).strip().split()).lower()
    if not text:
        return ""
    if isinstance(enum_vals, list):
        for enum_value in enum_vals:
            enum_text = str(enum_value).strip().lower()
            if text == enum_text:
                return enum_text
    return text


def _first_int(text: str) -> Optional[int]:
    match = re.search(r"-?\d+", text.replace(",", ""))
    if not match:
        return None
    return int(match.group(0))


def _is_informative_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        cleaned = value.strip().lower()
        if cleaned in {"", "unknown", "none", "null", "n/a", "na"}:
            return False
        return True
    return True


def _load_manifest(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


if __name__ == "__main__":
    main()
