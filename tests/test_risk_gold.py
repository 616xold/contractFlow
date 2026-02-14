from pathlib import Path

from scripts.evaluate_risk_gold import evaluate_risk_gold


def test_risk_gold_dataset_is_balanced() -> None:
    summary = evaluate_risk_gold(
        dataset_path=Path("data/risk_gold/risk_gold_v1.json"),
        model="gpt-5.2",
        enable_judge=False,
        judge_model=None,
        structured_outputs=True,
        use_default_field_meta=True,
        bins=5,
    )
    assert summary["cases_total"] == 15
    assert summary["class_balance"] == {"low": 5, "medium": 5, "high": 5}
    assert summary["balanced"] is True
    assert summary["accuracy"] >= 0.9
