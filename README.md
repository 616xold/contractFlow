# ContractFlow

ContractFlow is a baseline NDA/commercial contract extractor. It reads PDFs, extracts a fixed set of fields into JSON, and captures a simple risk summary. The goal is to evolve into a production-style, agentic pipeline with evaluation and UI, but today the focus is the Week-1 baseline CLI and plumbing.

## What’s implemented now
- **Baseline extractor** (`contractflow/core/extractor.py`): single-call LLM extraction, structured outputs (Pydantic + `responses.parse`), robust JSON recovery, schema validation/coercion, prompt-safety guards, and null-handling for nullable fields.
- **Schema & model** (`contractflow/schemas/contract_schema.json`, `contractflow/schemas/models.py`): fixed fields for NDAs/MSAs; stricter types (e.g., `effective_date` as ISO date), non-empty party names/governing law, enums, nullable ints, and “unknown” reserved for `data_transfer_outside_uk_eu`.
- **CLI tools**:
  - `scripts/baseline_extract.py`: run on a single PDF, saves parsed JSON and raw model output to `data/preds/`, prints token usage, supports strict/lenient validation and structured-outputs toggle.
  - `scripts/bulk_extract.py`: iterate over `data/raw_pdfs/*.pdf`, save preds/raws, and write a summary CSV with success/failure and token counts.
  - `scripts/download_samples.py`: fetches sample NDAs/MSAs into `data/raw_pdfs/`.
- **PDF text**: `contractflow/core/pdf_utils.py` for basic text extraction via `pypdf`.
- **Sample output**: `data/preds/nda_harvard.pred.json` demonstrates the pipeline; validation recorded an issue for missing `effective_date`.

## Repo layout
- `contractflow/core/` — extractor and PDF utilities.
- `contractflow/schemas/` — JSON schema + Pydantic model.
- `scripts/` — CLIs for single/bulk extraction and sample download.
- `data/raw_pdfs/` — sample PDFs (after running `download_samples`).
- `data/preds/` — predictions and raw model outputs.
- `docs/domain.md` — field definitions and initial risk heuristics.

## Setup
```bash
python -m venv .venv
. .venv/Scripts/activate  # PowerShell: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```
Add your key in `.env`:
```
OPENAI_API_KEY=sk-...
```

## Usage
Single PDF:
```bash
python scripts/baseline_extract.py data/raw_pdfs/nda_harvard.pdf --model gpt-5.2
```
- Outputs: `data/preds/nda_harvard.pred.json` (parsed + `_meta` with tokens/issues) and `data/preds/nda_harvard.raw.txt`.
- Defaults: `strict=False` (lenient), validation on, coercion on, structured outputs on.
- Turn on fail-fast: `--strict`
- Disable structured outputs: `--no-structured-outputs`
- Disable validation/coercion: `--no-validate`, `--no-coerce`

Bulk over a folder:
```bash
python scripts/bulk_extract.py --model gpt-5.2
```
Writes preds/raws per PDF plus `data/preds/summary.csv`.

Download sample PDFs:
```bash
python scripts/download_samples.py
```

## Current progress (Baseline)
- Prompt is guarded (clear delimiters, injection warnings) and enforces: return all keys, use `null` for nullable missing, `unknown` only for data_transfer_outside_uk_eu.
- Validation catches missing/extra keys, wrong types, enum drift, empty required strings, and disallows `"unknown"` for non-reserved fields. Lenient mode returns issues instead of raising.
- Structured outputs (`responses.parse` + Pydantic) for stronger adherence; falls back to JSON parsing with recovery.
- Token usage captured; `_meta` persisted with run settings and issues to aid eval/debug.

## Known gaps / next steps
- Improve recall for `effective_date` (observed null in `nda_harvard.pred.json`); consider regex/date heuristics or targeted field prompts.
- Add evaluation scripts against gold labels (`data/labels/`) once labels exist.
- Add chunking/RAG + per-field retrieval for better accuracy on longer contracts.
- Add risk post-processing using `docs/domain.md` heuristics as a secondary check.
- Add tests for extraction/validation and CI wiring.
- UI and integrations (CSV/Sheets/Notion) are planned but not yet started.

## Notes
- Default model is `gpt-5.2` with `reasoning={"effort": "none"}` and `temperature=0` for determinism/cost control.
- Strict mode will raise on any schema violation; lenient mode will keep `issues` in output and set offending fields to `null`.
