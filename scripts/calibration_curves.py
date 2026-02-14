"""Compute calibration curves for field confidence and risk confidence."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

# Ensure repo root is on PYTHONPATH when running as a script.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.evaluate import evaluate_predictions


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute calibration metrics from predictions vs labels.")
    parser.add_argument(
        "--preds-dir",
        type=Path,
        required=True,
        help="Directory containing .pred.json files.",
    )
    parser.add_argument(
        "--labels-dir",
        type=Path,
        default=REPO_ROOT / "data" / "labels",
        help="Directory containing label files.",
    )
    parser.add_argument(
        "--label-suffix",
        type=str,
        default=".gold.json",
        help="Label filename suffix (default: .gold.json).",
    )
    parser.add_argument(
        "--schema",
        type=Path,
        default=REPO_ROOT / "contractflow" / "schemas" / "contract_schema.json",
        help="Schema path used for normalization in evaluate.py.",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=10,
        help="Number of equal-width confidence bins (default: 10).",
    )
    parser.add_argument(
        "--exclude-derived",
        action="store_true",
        help="Exclude schema fields marked derived=true from field calibration.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional JSON output path.",
    )
    parser.add_argument(
        "--csv-dir",
        type=Path,
        default=None,
        help="Optional directory to write calibration CSVs.",
    )
    args = parser.parse_args()

    if args.bins < 2:
        raise ValueError("--bins must be >= 2")

    include_derived = not args.exclude_derived
    summary = evaluate_predictions(
        labels_dir=args.labels_dir,
        preds_dir=args.preds_dir,
        schema_path=args.schema,
        label_suffix=args.label_suffix,
        partial_threshold=0.85,
        bootstrap_samples=0,
        bootstrap_seed=42,
        include_derived=include_derived,
    )
    report = build_calibration_report(
        summary=summary,
        bins=args.bins,
    )

    print(json.dumps(report, indent=2))

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    if args.csv_dir:
        write_calibration_csvs(report, args.csv_dir)


def build_calibration_report(
    *,
    summary: Dict[str, Any],
    bins: int,
) -> Dict[str, Any]:
    field_points: Dict[str, list[Tuple[float, float]]] = {}
    overall_field_points: list[Tuple[float, float]] = []
    risk_points: list[Tuple[float, float]] = []

    for doc in summary.get("docs", []):
        if doc.get("status") != "evaluated":
            continue
        pred_path = doc.get("pred_path")
        if not isinstance(pred_path, str):
            continue
        payload = _load_json(Path(pred_path))
        if payload is None:
            continue
        retrieval = ((payload.get("_meta") or {}).get("retrieval") or {})
        fields_meta = retrieval.get("fields") if isinstance(retrieval, dict) else None
        fields_meta = fields_meta if isinstance(fields_meta, dict) else {}

        doc_fields = doc.get("fields")
        if not isinstance(doc_fields, dict):
            continue

        for field, field_eval in doc_fields.items():
            if not isinstance(field_eval, dict):
                continue
            is_exact = field_eval.get("exact")
            if not isinstance(is_exact, bool):
                continue
            meta = fields_meta.get(field)
            if not isinstance(meta, dict):
                continue
            confidence = _to_confidence(meta.get("confidence"))
            if confidence is None:
                continue
            point = (confidence, 1.0 if is_exact else 0.0)
            field_points.setdefault(field, []).append(point)
            overall_field_points.append(point)

        risk_meta = retrieval.get("risk") if isinstance(retrieval, dict) else None
        risk_meta = risk_meta if isinstance(risk_meta, dict) else {}
        risk_conf = _to_confidence(risk_meta.get("confidence"))
        risk_eval = doc_fields.get("risk_level") if isinstance(doc_fields, dict) else None
        risk_exact = risk_eval.get("exact") if isinstance(risk_eval, dict) else None
        if risk_conf is not None and isinstance(risk_exact, bool):
            risk_points.append((risk_conf, 1.0 if risk_exact else 0.0))

    per_field_report = {
        field: _compute_reliability(points, bins=bins)
        for field, points in sorted(field_points.items())
    }
    overall_field_report = _compute_reliability(overall_field_points, bins=bins)
    risk_report = _compute_reliability(risk_points, bins=bins)

    return {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "bins": bins,
        "docs_total": summary.get("docs_total"),
        "docs_evaluated": summary.get("docs_evaluated"),
        "preds_dir": _infer_preds_dir(summary),
        "overall_field_confidence": overall_field_report,
        "risk_confidence": risk_report,
        "per_field_confidence": per_field_report,
    }


def write_calibration_csvs(report: Dict[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    _write_bins_csv(
        bins=(((report.get("overall_field_confidence") or {}).get("bins")) or []),
        out_path=out_dir / "field_overall_bins.csv",
    )
    _write_bins_csv(
        bins=(((report.get("risk_confidence") or {}).get("bins")) or []),
        out_path=out_dir / "risk_bins.csv",
    )

    field_summary_path = out_dir / "field_calibration_summary.csv"
    with field_summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "field",
                "n",
                "accuracy",
                "avg_confidence",
                "ece",
                "mce",
                "brier",
            ],
        )
        writer.writeheader()
        for field, data in sorted((report.get("per_field_confidence") or {}).items()):
            if not isinstance(data, dict):
                continue
            writer.writerow(
                {
                    "field": field,
                    "n": data.get("n"),
                    "accuracy": data.get("accuracy"),
                    "avg_confidence": data.get("avg_confidence"),
                    "ece": data.get("ece"),
                    "mce": data.get("mce"),
                    "brier": data.get("brier"),
                }
            )


def _write_bins_csv(*, bins: list[Dict[str, Any]], out_path: Path) -> None:
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["bin_idx", "lower", "upper", "count", "avg_confidence", "accuracy", "gap"],
        )
        writer.writeheader()
        for item in bins:
            writer.writerow(item)


def _infer_preds_dir(summary: Dict[str, Any]) -> Optional[str]:
    docs = summary.get("docs")
    if not isinstance(docs, list):
        return None
    for doc in docs:
        if not isinstance(doc, dict):
            continue
        pred_path = doc.get("pred_path")
        if isinstance(pred_path, str) and pred_path:
            return str(Path(pred_path).parent)
    return None


def _to_confidence(value: Any) -> Optional[float]:
    if not isinstance(value, (int, float)):
        return None
    out = float(value)
    if out < 0.0:
        out = 0.0
    if out > 1.0:
        out = 1.0
    return out


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
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
        index = min(int(conf * bins), bins - 1)
        b = bucket[index]
        b["count"] += 1
        b["conf_sum"] += conf
        b["correct_sum"] += correct
        brier_sum += (conf - correct) ** 2
        conf_total += conf
        correct_total += correct

    ece = 0.0
    mce = 0.0
    out_bins: list[Dict[str, Any]] = []
    for item in bucket:
        count = int(item["count"])
        if count == 0:
            out_bins.append(
                {
                    "bin_idx": item["bin_idx"],
                    "lower": item["lower"],
                    "upper": item["upper"],
                    "count": 0,
                    "avg_confidence": None,
                    "accuracy": None,
                    "gap": None,
                }
            )
            continue
        avg_conf = item["conf_sum"] / count
        accuracy = item["correct_sum"] / count
        gap = abs(avg_conf - accuracy)
        ece += (count / total) * gap
        mce = max(mce, gap)
        out_bins.append(
            {
                "bin_idx": item["bin_idx"],
                "lower": item["lower"],
                "upper": item["upper"],
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
        "bins": out_bins,
    }


if __name__ == "__main__":
    main()
