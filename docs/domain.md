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

## Risk Rules (initial heuristic)

- Start with `medium`.
- Upgrade to `high` if:
  - liability is uncapped OR
  - governing law is outside UK/EU OR
  - data transfer is "yes" and there are no clear safeguards.
- Downgrade to `low` if:
  - liability is capped at a reasonable level (e.g. <= 12 months' fees) AND
  - governing_law is England and Wales (or similar) AND
  - term_length is 12 months or less.
