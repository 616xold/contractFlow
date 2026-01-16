# Labels

Gold labels live in this folder and are named like `<doc_stem>.gold.json`.

Guidelines:
- Use the schema in `contractflow/schemas/contract_schema.json`.
- Populate every field (use null only for nullable fields).
- Keep party names as they appear in the contract preamble.
- Normalize `effective_date` to ISO `YYYY-MM-DD` when possible.
- Store `term_length` in months (convert years to months).
- For `data_transfer_outside_uk_eu`, use only `yes`, `no`, or `unknown`.
