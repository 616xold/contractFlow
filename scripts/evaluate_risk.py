"""Evaluate risk classification outputs with confusion matrix and agreement metrics."""

from __future__ import annotations

import argparse
import json
import sys
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict

# Ensure repo root is on PYTHONPATH when running as a script.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


RISK_LEVELS = ["low", "medium", "high"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate risk classification predictions.")
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
        "--out",
        type=Path,
        default=None,
        help="Optional path to write JSON summary",
    )
    args = parser.parse_args()

    summary = evaluate_risk_predictions(
        labels_dir=args.labels_dir,
        preds_dir=args.preds_dir,
        label_suffix=args.label_suffix,
    )
    _print_summary(summary)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def evaluate_risk_predictions(
    *,
    labels_dir: Path,
    preds_dir: Path,
    label_suffix: str = ".gold.json",
) -> Dict[str, Any]:
    label_paths = sorted(labels_dir.glob(f"*{label_suffix}"))
    if not label_paths:
        raise ValueError(f"No label files found in {labels_dir} with suffix {label_suffix}")

    confusion = {gold: {pred: 0 for pred in RISK_LEVELS} for gold in RISK_LEVELS}
    docs: list[Dict[str, Any]] = []
    docs_total = len(label_paths)
    docs_evaluated = 0
    docs_missing_preds = 0
    exact_correct = 0
    explanation_similarities: list[float] = []

    arbitration_counts: Dict[str, int] = {}
    judge_available = 0
    judge_rule_agree = 0
    judge_final_agree = 0

    for label_path in label_paths:
        base = label_path.name[: -len(label_suffix)] if label_path.name.endswith(label_suffix) else label_path.stem
        pred_path = preds_dir / f"{base}.pred.json"
        if not pred_path.exists():
            docs_missing_preds += 1
            docs.append({"doc": base, "status": "missing_pred"})
            continue

        gold = _load_json(label_path)
        pred = _load_json(pred_path)
        gold_level = _normalize_level(gold.get("risk_level"))
        pred_level = _normalize_level(pred.get("risk_level"))
        if gold_level in RISK_LEVELS and pred_level in RISK_LEVELS:
            confusion[gold_level][pred_level] += 1

        gold_expl = str(gold.get("risk_explanation") or "")
        pred_expl = str(pred.get("risk_explanation") or "")
        expl_sim = _text_similarity(gold_expl, pred_expl)
        explanation_similarities.append(expl_sim)

        is_exact = gold_level == pred_level
        if is_exact:
            exact_correct += 1
        docs_evaluated += 1

        risk_meta = ((pred.get("_meta") or {}).get("retrieval") or {}).get("risk") or {}
        arbitration = risk_meta.get("arbitration")
        if isinstance(arbitration, str) and arbitration:
            arbitration_counts[arbitration] = arbitration_counts.get(arbitration, 0) + 1
        rule_level = _normalize_level(risk_meta.get("rule_level"))
        judge_level = _normalize_level(risk_meta.get("judge_level"))
        final_level = _normalize_level(risk_meta.get("risk_level") or pred_level)
        if judge_level in RISK_LEVELS:
            judge_available += 1
            if rule_level == judge_level:
                judge_rule_agree += 1
            if final_level == judge_level:
                judge_final_agree += 1

        docs.append(
            {
                "doc": base,
                "status": "evaluated",
                "risk_level_gold": gold_level,
                "risk_level_pred": pred_level,
                "risk_level_exact": is_exact,
                "risk_explanation_similarity": round(expl_sim, 4),
                "risk_meta": {
                    "arbitration": arbitration,
                    "rule_level": rule_level,
                    "judge_level": judge_level if judge_level in RISK_LEVELS else None,
                    "final_level": final_level,
                },
            }
        )

    per_class = _per_class_metrics(confusion)
    macro_f1_all = sum(metrics["f1"] for metrics in per_class.values()) / max(1, len(per_class))
    supported = [metrics["f1"] for metrics in per_class.values() if metrics["support"] > 0]
    macro_f1_supported = sum(supported) / max(1, len(supported))

    return {
        "docs_total": docs_total,
        "docs_evaluated": docs_evaluated,
        "docs_missing_preds": docs_missing_preds,
        "risk_level_accuracy": round(exact_correct / max(1, docs_evaluated), 4),
        "risk_explanation_similarity_avg": round(
            sum(explanation_similarities) / max(1, len(explanation_similarities)),
            4,
        ),
        "macro_f1": round(macro_f1_all, 4),
        "macro_f1_supported": round(macro_f1_supported, 4),
        "confusion_matrix": confusion,
        "per_class": per_class,
        "arbitration_counts": arbitration_counts,
        "judge_metrics": {
            "judge_available_docs": judge_available,
            "judge_rule_agreement": round(judge_rule_agree / max(1, judge_available), 4),
            "judge_final_agreement": round(judge_final_agree / max(1, judge_available), 4),
        },
        "docs": docs,
    }


def _per_class_metrics(confusion: Dict[str, Dict[str, int]]) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for level in RISK_LEVELS:
        tp = confusion[level][level]
        fp = sum(confusion[gold][level] for gold in RISK_LEVELS if gold != level)
        fn = sum(confusion[level][pred] for pred in RISK_LEVELS if pred != level)
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)
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


def _text_similarity(a: str, b: str) -> float:
    norm_a = " ".join(a.lower().split())
    norm_b = " ".join(b.lower().split())
    if not norm_a and not norm_b:
        return 1.0
    if not norm_a or not norm_b:
        return 0.0
    return SequenceMatcher(None, norm_a, norm_b).ratio()


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def _print_summary(summary: Dict[str, Any]) -> None:
    print(f"docs_total={summary['docs_total']} docs_evaluated={summary['docs_evaluated']}")
    print(f"docs_missing_preds={summary['docs_missing_preds']}")
    print(f"risk_level_accuracy={summary['risk_level_accuracy']}")
    print(f"macro_f1={summary['macro_f1']}")
    print(f"macro_f1_supported={summary['macro_f1_supported']}")
    print(f"risk_explanation_similarity_avg={summary['risk_explanation_similarity_avg']}")
    print("per_class:")
    for level, metrics in summary["per_class"].items():
        print(
            f"  {level}: precision={metrics['precision']} recall={metrics['recall']} "
            f"f1={metrics['f1']} support={metrics['support']}"
        )
    if summary.get("arbitration_counts"):
        print("arbitration_counts:")
        for key, count in summary["arbitration_counts"].items():
            print(f"  {key}={count}")


if __name__ == "__main__":
    main()
