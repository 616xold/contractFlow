# Risk Gold Set

This folder stores balanced gold labels for the post-extraction risk classifier.

Current artifact:
- `risk_gold_v1.json`: 15 manually adjudicated structured contract profiles.
- Class balance: 5 `low`, 5 `medium`, 5 `high`.

Purpose:
- Evaluate risk orchestration quality independently from extraction quality.
- Track calibration and class-wise behavior of the risk engine and optional judge.

Notes:
- These are structured profiles (policy-focused cases), not raw PDF annotations.
- Use `scripts/evaluate_risk_gold.py` to score the risk engine against this set.
