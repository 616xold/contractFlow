from scripts.calibration_curves import _compute_reliability


def test_compute_reliability_metrics() -> None:
    points = [
        (0.9, 1.0),
        (0.8, 1.0),
        (0.2, 0.0),
        (0.1, 0.0),
    ]
    report = _compute_reliability(points, bins=2)
    assert report["n"] == 4
    assert report["accuracy"] == 0.5
    assert report["avg_confidence"] == 0.5
    assert report["ece"] == 0.15
    assert report["mce"] == 0.15
    assert report["brier"] == 0.025
    assert len(report["bins"]) == 2
