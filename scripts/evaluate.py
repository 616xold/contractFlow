"""Evaluate predictions against gold labels with evidence coverage metrics."""

from __future__ import annotations

import argparse
import json
import re
import sys
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, Tuple

# Ensure repo root is on PYTHONPATH when running as a script.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))



def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate predictions vs labels.")
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
        help="Path to the JSON schema describing fields to evaluate",
    )
    parser.add_argument(
        "--partial-threshold",
        type=float,
        default=0.85,
        help="Partial match threshold for string similarity (default: 0.85)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional path to write JSON summary",
    )
    args = parser.parse_args()

    summary = evaluate_predictions(
        labels_dir=args.labels_dir,
        preds_dir=args.preds_dir,
        schema_path=args.schema,
        label_suffix=args.label_suffix,
        partial_threshold=args.partial_threshold,
    )

    _print_summary(summary)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def evaluate_predictions(
    *,
    labels_dir: Path,
    preds_dir: Path,
    schema_path: Path,
    label_suffix: str = ".gold.json",
    partial_threshold: float = 0.85,
) -> Dict[str, Any]:
    schema = _load_json(schema_path)
    label_paths = sorted(labels_dir.glob(f"*{label_suffix}"))
    if not label_paths:
        raise ValueError(f"No label files found in {labels_dir} with suffix {label_suffix}")

    field_stats: Dict[str, Dict[str, Any]] = {
        field: {
            "exact_correct": 0,
            "partial_correct": 0,
            "total": 0,
            "avg_similarity": 0.0,
        }
        for field in schema.keys()
    }
    docs: list[Dict[str, Any]] = []

    evidence_ratios: list[float] = []
    hit_ratios: list[float] = []

    missing_preds = 0
    evaluated_docs = 0

    overall_exact = 0
    overall_partial = 0
    overall_total = 0

    for label_path in label_paths:
        base = label_path.name
        if base.endswith(label_suffix):
            base = base[: -len(label_suffix)]
        else:
            base = label_path.stem
        pred_path = preds_dir / f"{base}.pred.json"

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
        gold_fields = {k: v for k, v in gold.items() if k != "_meta"}

        exact_correct = 0
        partial_correct = 0
        total = 0
        per_field: Dict[str, Any] = {}

        for field, meta_def in schema.items():
            gold_value = gold_fields.get(field, None)
            pred_value = pred_fields.get(field, None)
            norm_gold = _normalize_value(gold_value, meta_def)
            norm_pred = _normalize_value(pred_value, meta_def)
            is_exact = norm_gold == norm_pred
            similarity = _field_similarity(norm_gold, norm_pred, meta_def)
            is_partial = is_exact or similarity >= partial_threshold

            total += 1
            if is_exact:
                exact_correct += 1
            if is_partial:
                partial_correct += 1

            field_stats[field]["total"] += 1
            if is_exact:
                field_stats[field]["exact_correct"] += 1
            if is_partial:
                field_stats[field]["partial_correct"] += 1
            field_stats[field]["avg_similarity"] += similarity

            per_field[field] = {
                "gold": norm_gold,
                "pred": norm_pred,
                "exact": is_exact,
                "partial": is_partial,
                "similarity": round(similarity, 4),
            }

        overall_exact += exact_correct
        overall_partial += partial_correct
        overall_total += total

        evaluated_docs += 1
        accuracy_exact = (exact_correct / total) if total else 0.0
        accuracy_partial = (partial_correct / total) if total else 0.0

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
                "fields_exact": exact_correct,
                "fields_partial": partial_correct,
                "fields_total": total,
                "accuracy_exact": round(accuracy_exact, 4),
                "accuracy_partial": round(accuracy_partial, 4),
                "coverage": coverage,
                "fields": per_field,
            }
        )

    field_accuracy = {}
    for field, stats in field_stats.items():
        total = stats["total"]
        exact = stats["exact_correct"]
        partial = stats["partial_correct"]
        avg_similarity = stats["avg_similarity"] / total if total else 0.0
        field_accuracy[field] = {
            "exact_correct": exact,
            "partial_correct": partial,
            "total": total,
            "accuracy_exact": round((exact / total) if total else 0.0, 4),
            "accuracy_partial": round((partial / total) if total else 0.0, 4),
            "avg_similarity": round(avg_similarity, 4),
        }

    summary = {
        "docs_total": len(label_paths),
        "docs_evaluated": evaluated_docs,
        "docs_missing_preds": missing_preds,
        "overall_accuracy_exact": round((overall_exact / overall_total) if overall_total else 0.0, 4),
        "overall_accuracy_partial": round((overall_partial / overall_total) if overall_total else 0.0, 4),
        "field_accuracy": field_accuracy,
        "evidence_coverage": {
            "avg_evidence_ratio": round(_mean(evidence_ratios), 4) if evidence_ratios else None,
            "avg_hit_ratio": round(_mean(hit_ratios), 4) if hit_ratios else None,
            "docs_with_evidence_ratio": len(evidence_ratios),
            "docs_with_hit_ratio": len(hit_ratios),
        },
        "docs": docs,
    }

    return summary


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


def _field_similarity(gold: Any, pred: Any, meta: Dict[str, Any]) -> float:
    expected = meta.get("type")
    enum_vals = meta.get("enum")
    if gold is None and pred is None:
        return 1.0
    if expected in {"integer", "boolean"}:
        return 1.0 if gold == pred else 0.0
    if enum_vals:
        return 1.0 if gold == pred else 0.0
    return _text_similarity(str(gold), str(pred))


def _text_similarity(a: str, b: str) -> float:
    norm_a = _normalize_text(a)
    norm_b = _normalize_text(b)
    if not norm_a and not norm_b:
        return 1.0
    if not norm_a or not norm_b:
        return 0.0
    return SequenceMatcher(None, norm_a, norm_b).ratio()


def _normalize_text(text: str) -> str:
    lowered = text.lower()
    lowered = re.sub(r"[^a-z0-9]+", " ", lowered)
    return " ".join(lowered.split())


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
    print(f"overall_accuracy_exact={summary['overall_accuracy_exact']}")
    print(f"overall_accuracy_partial={summary['overall_accuracy_partial']}")
    print("field_accuracy:")
    for field, stats in summary["field_accuracy"].items():
        print(
            f"  {field}: exact {stats['exact_correct']}/{stats['total']} "
            f"({stats['accuracy_exact']}), partial {stats['partial_correct']}/{stats['total']} "
            f"({stats['accuracy_partial']})"
        )

    coverage = summary.get("evidence_coverage", {})
    if coverage:
        print("evidence_coverage:")
        print(f"  avg_evidence_ratio={coverage.get('avg_evidence_ratio')}")
        print(f"  avg_hit_ratio={coverage.get('avg_hit_ratio')}")
        print(f"  docs_with_evidence_ratio={coverage.get('docs_with_evidence_ratio')}")
        print(f"  docs_with_hit_ratio={coverage.get('docs_with_hit_ratio')}")


if __name__ == "__main__":
    main()
