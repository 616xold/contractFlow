from contractflow.core.extractor import _should_cost_aware_early_exit


def test_cost_aware_early_exit_accepts_high_quality_snapshot() -> None:
    snapshot = {
        "avg_confidence": 0.9,
        "evidence_ratio": 0.9,
        "non_null_ratio": 1.0,
        "critical_min_confidence": 0.85,
        "disagreement_rate": 0.05,
        "conflict_fields": [],
        "critical_conflict_fields": [],
    }
    should_exit, reasons = _should_cost_aware_early_exit(
        snapshot,
        min_avg_confidence=0.82,
        min_evidence_ratio=0.7,
        min_non_null_ratio=0.9,
        min_critical_confidence=0.75,
        max_disagreement_rate=0.2,
    )
    assert should_exit is True
    assert reasons == []


def test_cost_aware_early_exit_rejects_when_thresholds_fail() -> None:
    snapshot = {
        "avg_confidence": 0.7,
        "evidence_ratio": 0.4,
        "non_null_ratio": 0.85,
        "critical_min_confidence": 0.6,
        "disagreement_rate": 0.4,
        "conflict_fields": ["liability_cap"],
        "critical_conflict_fields": ["liability_cap"],
    }
    should_exit, reasons = _should_cost_aware_early_exit(
        snapshot,
        min_avg_confidence=0.82,
        min_evidence_ratio=0.7,
        min_non_null_ratio=0.9,
        min_critical_confidence=0.75,
        max_disagreement_rate=0.2,
    )
    assert should_exit is False
    assert "avg_confidence_below_threshold" in reasons
    assert "evidence_ratio_below_threshold" in reasons
    assert "non_null_ratio_below_threshold" in reasons
    assert "critical_min_confidence_below_threshold" in reasons
    assert "disagreement_rate_above_threshold" in reasons
    assert "deterministic_conflicts_present" in reasons
    assert "critical_conflicts_present" in reasons
