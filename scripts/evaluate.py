"""Evaluate predictions against gold labels with evidence coverage metrics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

# Ensure repo root is on PYTHONPATH when running as a script.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))



def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate predictions vs gold labels.")
    parser.add_argument(
        "--preds-dir",
        type=Path,
        default=REPO_ROOT / "data" / "preds",
        help="Directory containing .pred.json files",
    )
    parser.add_argument(
        "--labels-dir",
        type=Path,
        default=REPO_ROOT / "data" / "labels",
        help="Directory containing .gold.json files",
    )
    parser.add_argument(
        "--schema",
        type=Path,
        default=REPO_ROOT / "contractflow" / "schemas" / "contract_schema.json",
        help="Path to the JSON schema describing fields to evaluate",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional path to write JSON summary",
    )
    args = parser.parse_args()

    schema = _load_json(args.schema)
    label_paths = sorted(args.labels_dir.glob("*.gold.json"))
    if not label_paths:
        print(f"No label files found in {args.labels_dir}", file=sys.stderr)
        raise SystemExit(1)

    field_stats: Dict[str, Dict[str, Any]] = {
        field: {"correct": 0, "total": 0} for field in schema.keys()
    }
    docs: list[Dict[str, Any]] = []

    evidence_ratios: list[float] = []
    hit_ratios: list[float] = []

    missing_preds = 0
    evaluated_docs = 0

    for label_path in label_paths:
        base = label_path.stem
        if base.endswith(".gold"):
            base = base[: -len(".gold")]
        pred_path = args.preds_dir / f"{base}.pred.json"

        gold = _load_json(label_path)
        pred: Dict[str, Any] = {}
        meta: Dict[str, Any] = {}

        if not pred_path.exists():
            missing_preds += 1
            docs.append(
                {
                    "doc": base,
                    "label_path": str(label_path),
                    "pred_path": None,
                    "status": "missing_pred",
                }
            )
            continue

        pred = _load_json(pred_path)
        meta = pred.get("_meta", {}) if isinstance(pred, dict) else {}
        pred_fields = {k: v for k, v in pred.items() if k != "_meta"}

        correct = 0
        total = 0
        per_field: Dict[str, Any] = {}

        for field, meta_def in schema.items():
            gold_value = gold.get(field, None)
            pred_value = pred_fields.get(field, None)
            norm_gold = _normalize_value(gold_value, meta_def)
            norm_pred = _normalize_value(pred_value, meta_def)
            is_correct = norm_gold == norm_pred

            total += 1
            if is_correct:
                correct += 1

            field_stats[field]["total"] += 1
            if is_correct:
                field_stats[field]["correct"] += 1

            per_field[field] = {
                "gold": norm_gold,
                "pred": norm_pred,
                "correct": is_correct,
            }

        evaluated_docs += 1
        accuracy = (correct / total) if total else 0.0

        coverage = (meta.get("retrieval") or {}).get("coverage") or {}
        evidence_ratio = coverage.get("evidence_ratio")
        hit_ratio = coverage.get("hit_ratio")
        if isinstance(evidence_ratio, (int, float)):
            evidence_ratios.append(float(evidence_ratio))
        if isinstance(hit_ratio, (int, float)):
            hit_ratios.append(float(hit_ratio))

        docs.append(
            {
                "doc": base,
                "label_path": str(label_path),
                "pred_path": str(pred_path),
                "status": "evaluated",
                "fields_correct": correct,
                "fields_total": total,
                "accuracy": round(accuracy, 4),
                "coverage": coverage,
                "fields": per_field,
            }
        )

    field_accuracy = {}
    overall_correct = 0
    overall_total = 0

    for field, stats in field_stats.items():
        total = stats["total"]
        correct = stats["correct"]
        overall_correct += correct
        overall_total += total
        field_accuracy[field] = {
            "correct": correct,
            "total": total,
            "accuracy": round((correct / total) if total else 0.0, 4),
        }

    summary = {
        "docs_total": len(label_paths),
        "docs_evaluated": evaluated_docs,
        "docs_missing_preds": missing_preds,
        "overall_accuracy": round((overall_correct / overall_total) if overall_total else 0.0, 4),
        "field_accuracy": field_accuracy,
        "evidence_coverage": {
            "avg_evidence_ratio": round(_mean(evidence_ratios), 4) if evidence_ratios else None,
            "avg_hit_ratio": round(_mean(hit_ratios), 4) if hit_ratios else None,
            "docs_with_evidence_ratio": len(evidence_ratios),
            "docs_with_hit_ratio": len(hit_ratios),
        },
        "docs": docs,
    }

    _print_summary(summary)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return data


def _normalize_value(value: Any, meta: Dict[str, Any]) -> Any:
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
            match = _first_int(value)
            return match
        return value

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
        return value

    text = str(value).strip()
    text = " ".join(text.split())
    text_lower = text.lower()
    if enum_vals:
        for enum_value in enum_vals:
            if text_lower == str(enum_value).strip().lower():
                return str(enum_value)
    return text_lower


def _first_int(text: str) -> Any:
    match = None
    for token in text.replace(",", " ").split():
        if token.lstrip("-").isdigit():
            match = int(token)
            break
    return match


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _print_summary(summary: Dict[str, Any]) -> None:
    print(f"docs_total={summary['docs_total']} docs_evaluated={summary['docs_evaluated']}")
    print(f"docs_missing_preds={summary['docs_missing_preds']}")
    print(f"overall_accuracy={summary['overall_accuracy']}")
    print("field_accuracy:")
    for field, stats in summary["field_accuracy"].items():
        print(f"  {field}: {stats['correct']}/{stats['total']} ({stats['accuracy']})")

    coverage = summary.get("evidence_coverage", {})
    if coverage:
        print("evidence_coverage:")
        print(f"  avg_evidence_ratio={coverage.get('avg_evidence_ratio')}")
        print(f"  avg_hit_ratio={coverage.get('avg_hit_ratio')}")
        print(f"  docs_with_evidence_ratio={coverage.get('docs_with_evidence_ratio')}")
        print(f"  docs_with_hit_ratio={coverage.get('docs_with_hit_ratio')}")


if __name__ == "__main__":
    main()
