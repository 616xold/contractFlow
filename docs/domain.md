# ContractFlow - Domain Notes

ContractFlow targets NDAs and simple SaaS/commercial agreements between a vendor and a customer.

## Extraction Fields

- `doc_type`: `nda | msa | other`
- `party_a_name` / `party_b_name`: legal party names from preamble/signature areas
- `effective_date`: contract effective date (ISO format when possible)
- `term_length`: initial term length in months
- `governing_law`: governing jurisdiction
- `termination_notice_days`: notice days for termination for convenience
- `liability_cap`: liability cap clause text (normalized string)
- `non_solicit_clause_present`: boolean flag
- `data_transfer_outside_uk_eu`: `yes | no | unknown`

## Risk Outputs

- `risk_level` and `risk_explanation` are **derived fields**.
- They are not extracted directly from the contract text prompt.
- They are produced only after extraction by the post-extraction risk orchestrator.

## Risk Engine V2

Implemented in `contractflow/core/risk_engine.py` with policy config in `docs/risk_policy.json`.

- Output classes: `low`, `medium`, `high`
- Deterministic weighted factors:
  - liability cap posture (uncapped/unknown/capped ranges)
  - governing law region (UK/EU vs outside vs unknown)
  - cross-border data transfer posture
  - term length
  - termination notice period
  - non-solicit presence
- Hard-trigger floors for high-risk combinations.
- Uncertainty-aware adjustment using evidence/confidence coverage.
- Optional LLM risk judge for arbitration over deterministic output.

## Post-Extraction Risk Orchestration

Implemented in `contractflow/core/extractor.py` and controlled by `risk_orchestration` in `docs/risk_policy.json`.

- Stage order:
  1. Run deterministic rules-only risk scoring on extracted values.
  2. Check trigger conditions (critical unknowns, low critical confidence, high uncertainty).
  3. If triggered, run a risk-review agent over targeted retrieved evidence for risk-input fields only.
  4. Apply safe typed corrections to risk-input fields.
  5. Recompute final risk (rules + optional risk judge).
- Full trace is persisted in `_meta.retrieval.risk.orchestration`:
  - trigger reasons
  - review rounds
  - applied corrections
  - before/after risk-input snapshots
  - token usage for risk review

This keeps risk explainable and deterministic-first while adding agentic recovery where extraction uncertainty is high.

## Liability Cap Reliability (Extraction + Eval)

- Shared parser in `contractflow/core/liability.py` extracts structured liability signals:
  - uncapped posture
  - fee-window caps in months (e.g., 12 months, 1 year -> 12 months)
  - fixed monetary caps with currency (when explicit)
- Extraction normalization in `contractflow/core/extractor.py` canonicalizes `liability_cap` to stable forms:
  - `uncapped`
  - `<N> months fees`
  - `<CUR> <amount>`
- Evaluation in `scripts/evaluate.py` uses semantic liability similarity instead of plain string overlap for `liability_cap`.
  This reduces false negatives where wording differs but cap semantics are equivalent.
