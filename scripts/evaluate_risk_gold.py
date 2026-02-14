"""Evaluate risk_engine against the balanced risk-gold profile set."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

# Ensure repo root is on PYTHONPATH when running as a script.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from contractflow.core.extractor import DEFAULT_MODEL
from contractflow.core.risk_engine import assess_contract_risk

RISK_LEVELS = ("low", "medium", "high")
_RISK_FIELDS = (
    "liability_cap",
    "governing_law",
    "data_transfer_outside_uk_eu",
    "term_length",
    "termination_notice_days",
    "non_solicit_clause_present",
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate policy risk model on balanced risk-gold profiles.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=REPO_ROOT / "data" / "risk_gold" / "risk_gold_v1.json",
        help="Path to risk-gold JSON dataset.",
    )
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Model name used by risk engine.")
    parser.add_argument(
        "--enable-judge",
        action="store_true",
        help="Enable judge arbitration while evaluating (defaults to rules-only).",
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        default=None,
        help="Optional judge model override (defaults to --model).",
    )
    parser.add_argument(
        "--no-structured-outputs",
        action="store_true",
        help="Disable structured output parsing for judge calls.",
    )
    parser.add_argument(
        "--no-default-field-meta",
        action="store_true",
        help="Do not synthesize high-coverage field metadata for profiles that omit field_meta.",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=10,
        help="Number of bins for risk-confidence calibration output.",
    )
    parser.add_argument("--out", type=Path, default=None, help="Optional JSON output path.")
    args = parser.parse_args()

    summary = evaluate_risk_gold(
        dataset_path=args.dataset,
        model=args.model,
        enable_judge=args.enable_judge,
        judge_model=args.judge_model,
        structured_outputs=not args.no_structured_outputs,
        use_default_field_meta=not args.no_default_field_meta,
        bins=args.bins,
    )
    print(json.dumps(summary, indent=2))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def evaluate_risk_gold(
    *,
    dataset_path: Path,
    model: str,
    enable_judge: bool,
    judge_model: str | None,
    structured_outputs: bool,
    use_default_field_meta: bool,
    bins: int,
) -> Dict[str, Any]:
    if bins < 2:
        raise ValueError("bins must be >= 2")

    payload = _load_json(dataset_path)
    cases = payload.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError(f"Dataset has no cases: {dataset_path}")

    class_balance = {label: 0 for label in RISK_LEVELS}
    confusion = {gold: {pred: 0 for pred in RISK_LEVELS} for gold in RISK_LEVELS}
    confidence_points: list[Tuple[float, float]] = []
    details: list[Dict[str, Any]] = []
    exact = 0

    for idx, case in enumerate(cases):
        if not isinstance(case, dict):
            continue
        case_id = str(case.get("id") or f"case_{idx:03d}")
        gold = _normalize_level(case.get("gold_risk_level"))
        if gold not in RISK_LEVELS:
            raise ValueError(f"Invalid gold_risk_level for case {case_id}: {case.get('gold_risk_level')}")
        class_balance[gold] += 1

        values = case.get("values")
        if not isinstance(values, dict):
            raise ValueError(f"Case {case_id} has no valid values object")

        field_meta = case.get("field_meta")
        if not isinstance(field_meta, dict):
            field_meta = _default_field_meta(values) if use_default_field_meta else None

        assessment = assess_contract_risk(
            values=values,
            field_meta=field_meta,
            model=model,
            structured_outputs=structured_outputs,
            enable_judge=enable_judge,
            judge_model=judge_model,
        )
        pred = _normalize_level(assessment.risk_level)
        if pred not in RISK_LEVELS:
            pred = "high"

        confusion[gold][pred] += 1
        is_exact = pred == gold
        if is_exact:
            exact += 1
        confidence_points.append((float(assessment.confidence), 1.0 if is_exact else 0.0))

        details.append(
            {
                "id": case_id,
                "gold_risk_level": gold,
                "pred_risk_level": pred,
                "correct": is_exact,
                "confidence": round(float(assessment.confidence), 4),
                "score": round(float(assessment.score), 4),
                "rule_level": assessment.rule_level,
                "rule_score": round(float(assessment.rule_score), 4),
                "arbitration": assessment.arbitration,
                "hard_triggers": list(assessment.hard_triggers),
                "notes": case.get("notes"),
            }
        )

    total = sum(class_balance.values())
    per_class = _per_class_metrics(confusion)
    calibration = _compute_reliability(confidence_points, bins=bins)

    counts = list(class_balance.values())
    balanced = bool(counts) and (max(counts) - min(counts) <= 1)
    return {
        "dataset": str(dataset_path),
        "dataset_version": payload.get("version"),
        "labeling_protocol": payload.get("labeling_protocol"),
        "model": model,
        "enable_judge": bool(enable_judge),
        "judge_model": judge_model or model,
        "structured_outputs": bool(structured_outputs),
        "use_default_field_meta": bool(use_default_field_meta),
        "cases_total": total,
        "accuracy": round(exact / max(1, total), 4),
        "class_balance": class_balance,
        "balanced": balanced,
        "confusion_matrix": confusion,
        "per_class": per_class,
        "confidence_calibration": calibration,
        "cases": details,
    }


def _default_field_meta(values: Dict[str, Any]) -> Dict[str, Any]:
    meta: Dict[str, Any] = {}
    for field in _RISK_FIELDS:
        value = values.get(field)
        has_value = value is not None and str(value).strip().lower() not in {"", "null", "none", "unknown"}
        confidence = 0.9 if has_value else 0.62
        snippet_text = "unknown" if value is None else str(value)
        meta[field] = {
            "confidence": confidence,
            "evidence": [
                {
                    "page_num": 1,
                    "heading": "risk profile",
                    "snippet": snippet_text,
                }
            ],
        }
    return meta


def _per_class_metrics(confusion: Dict[str, Dict[str, int]]) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for level in RISK_LEVELS:
        tp = confusion[level][level]
        fp = sum(confusion[gold][level] for gold in RISK_LEVELS if gold != level)
        fn = sum(confusion[level][pred] for pred in RISK_LEVELS if pred != level)
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
        out[level] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": int(sum(confusion[level].values())),
        }
    return out


def _normalize_level(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text in RISK_LEVELS:
        return text
    return "unknown"


def _load_json(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def _compute_reliability(points: Iterable[Tuple[float, float]], *, bins: int) -> Dict[str, Any]:
    values = list(points)
    total = len(values)
    if total == 0:
        return {
            "n": 0,
            "accuracy": None,
            "avg_confidence": None,
            "ece": None,
            "mce": None,
            "brier": None,
            "bins": [],
        }

    bucket = [
        {
            "bin_idx": idx,
            "lower": round(idx / bins, 4),
            "upper": round((idx + 1) / bins, 4),
            "count": 0,
            "conf_sum": 0.0,
            "correct_sum": 0.0,
        }
        for idx in range(bins)
    ]
    brier_sum = 0.0
    conf_total = 0.0
    correct_total = 0.0
    for conf, correct in values:
        conf = max(0.0, min(1.0, float(conf)))
        correct = max(0.0, min(1.0, float(correct)))
        index = min(int(conf * bins), bins - 1)
        row = bucket[index]
        row["count"] += 1
        row["conf_sum"] += conf
        row["correct_sum"] += correct
        conf_total += conf
        correct_total += correct
        brier_sum += (conf - correct) ** 2

    ece = 0.0
    mce = 0.0
    rows: list[Dict[str, Any]] = []
    for row in bucket:
        count = int(row["count"])
        if count == 0:
            rows.append(
                {
                    "bin_idx": row["bin_idx"],
                    "lower": row["lower"],
                    "upper": row["upper"],
                    "count": 0,
                    "avg_confidence": None,
                    "accuracy": None,
                    "gap": None,
                }
            )
            continue
        avg_conf = row["conf_sum"] / count
        accuracy = row["correct_sum"] / count
        gap = abs(avg_conf - accuracy)
        ece += (count / total) * gap
        mce = max(mce, gap)
        rows.append(
            {
                "bin_idx": row["bin_idx"],
                "lower": row["lower"],
                "upper": row["upper"],
                "count": count,
                "avg_confidence": round(avg_conf, 4),
                "accuracy": round(accuracy, 4),
                "gap": round(gap, 4),
            }
        )

    return {
        "n": total,
        "accuracy": round(correct_total / total, 4),
        "avg_confidence": round(conf_total / total, 4),
        "ece": round(ece, 4),
        "mce": round(mce, 4),
        "brier": round(brier_sum / total, 4),
        "bins": rows,
    }


if __name__ == "__main__":
    main()
