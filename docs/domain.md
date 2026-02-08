# ContractFlow – Domain Notes

We focus on NDAs and simple SaaS / commercial contracts between a small vendor and a customer.

## Fields

- `party_a_name` / `party_b_name`: Full legal names of the parties listed in the preamble.
- `effective_date`: The date the agreement comes into force (often in the first paragraph or signature block).
- `term_length`: Number of months/years the agreement is stated to last (ignore auto-renewal for now, just capture initial term).
- `governing_law`: Jurisdiction specified in the "Governing Law" or "Law and Jurisdiction" clause.
- `termination_notice_days`: If the contract allows termination for convenience, capture the required notice period.
- `liability_cap`: Text describing the cap on liability (e.g. "12 months of Fees").
- `non_solicit_clause_present`: True if there's any clause restricting solicitation of staff/customers.
- `data_transfer_outside_uk_eu`: Yes if the agreement clearly allows transfers of personal data outside the UK/EU, otherwise "no" or "unknown".

## Risk Rules (v2)

ContractFlow now uses a policy-driven risk engine (`contractflow/core/risk_engine.py`) with a
versioned policy file (`docs/risk_policy.json`).

- Output classes remain: `low`, `medium`, `high`.
- Core factors:
  - liability cap quality (uncapped/unknown/capped range)
  - governing law region (UK/EU vs outside, with unknown as an explicit bucket)
  - cross-border transfer posture (`yes`/`no`/`unknown`)
  - term length
  - termination notice period
  - non-solicit protection
- Scoring:
  - weighted additive rule score with uncertainty adjustment based on evidence/confidence coverage.
  - hard-trigger floors for high-risk combinations (e.g. uncapped liability + outside law).
- Optional judge pass:
  - an LLM judge reviews only structured factors + uncertainty metadata.
  - arbitration policy decides whether to keep rule output or accept judge override.

This keeps risk explainable (deterministic factors and weights) while adding agentic robustness
through judge-based arbitration.
