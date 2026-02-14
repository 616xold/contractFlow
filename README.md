# ContractFlow

Agentic contract extraction from PDF with retrieval, field-level specialists, verifier loops, and auditable post-extraction risk orchestration.

This project is built as a portfolio-grade AI engineering system: not just "one prompt", but a measurable multi-agent pipeline with explicit evidence, confidence, arbitration, and ablations.

## Why This Project

Contract extraction pipelines fail in different ways:

- one-shot prompts miss clause-level details
- retrieval alone can reduce context cost but still mis-extract fields
- high-stakes outputs need verification and deterministic guardrails

ContractFlow addresses this with staged agentic execution and evaluation-first development.

## What Makes It Agentic

1. **Retriever agent**
- Chunks contracts by page/section heading and returns top-k evidence chunks.

2. **Field agents**
- One agent per schema field.
- Each returns `value + evidence snippets + confidence + issues`.

3. **Verifier/Judge agent**
- Decides `accept`, `revise`, or `unknown`.
- Can trigger targeted retrieval + repair passes.

4. **Post-extraction risk orchestrator (v2)**
- Deterministic policy score first, then optional risk-review agent and judge arbitration.
- Full factor trace persisted in `_meta.retrieval.risk`.

## Architecture

```mermaid
flowchart LR
    A[PDF] --> B[Text + OCR]
    B --> C[Chunk by page/heading]
    C --> D[Retriever]
    D --> E[Global baseline]
    D --> F[Field agents]
    F --> G[Candidate select]
    G --> H[Verifier/Judge<br/>accept, revise, unknown]
    H -->|revise| D
    H --> I[Normalize + validate]
    I --> J[Risk Orchestrator V2<br/>rules, review, judge]
    J --> K[JSON + audit]
```

## Benchmark Snapshot (Gold Labels, 5 CUAD Docs)

Benchmark date: **February 14, 2026**  
Canonical artifact: `data/benchmarks/gold_ablation_presentation.json`

| Mode | Exact Accuracy | Partial Accuracy | Exact CI95 | Avg Total Tokens / Doc | Delta Exact vs Naive |
|---|---:|---:|---:|---:|---:|
| naive | 0.8500 | 0.9333 | 0.8333..0.8833 | 12,692 | +0.0000 |
| retrieval | 0.8000 | 0.8333 | 0.7667..0.8333 | 12,609.8 | -0.0500 |
| field_agents | 0.7833 | 0.8500 | 0.7000..0.8667 | 29,178 | -0.0667 |
| orchestrated (tuned) | **0.9167** | **0.9167** | **0.8667..0.9667** | **26,178.8** | **+0.0667** |

Notes:
- Gold set currently contains 5 labeled CUAD contracts (`.gold.json`).
- This benchmark includes derived fields (`risk_level`, `risk_explanation`) for all modes.
- Orchestrated uses a tuned low-token profile (see optimization table below).

### Accuracy Diagram

```mermaid
xychart-beta
    title "Exact Acc (gold-5)"
    x-axis ["N", "R", "F", "O"]
    y-axis "acc" 0 --> 1
    bar [0.85, 0.8, 0.7833, 0.9167]
```
`N=naive`, `R=retrieval`, `F=field_agents`, `O=orchestrated`

### Token Usage Diagram

```mermaid
xychart-beta
    title "Avg Tokens/Doc (gold-5)"
    x-axis ["N", "R", "F", "O"]
    y-axis "tokens" 0 --> 32000
    bar [12692, 12609.8, 29178, 26178.8]
```
`N=naive`, `R=retrieval`, `F=field_agents`, `O=orchestrated`

### Orchestrated Token Optimization (Gold-5)

| Orchestrated Profile | Exact | Partial | Avg Tokens / Doc |
|---|---:|---:|---:|
| default | 0.8500 | 0.8500 | 53,863.6 |
| tuned | **0.9167** | **0.9167** | **26,178.8** |

Tuned settings used in this benchmark:
- `top_k=2`
- `max_chunk_chars=800`
- `chunk_max_chars=1300`
- `max_repairs=2`
- `disable_verifier=true`
- `risk_review_top_k=2`

## Field-Level Signal (Orchestrated, Gold-5 Exact Accuracy)

- Strong fields:
  - `doc_type`: 1.00
  - `party_a_name`: 1.00
  - `governing_law`: 1.00
  - `liability_cap`: 1.00
  - `risk_level`: 1.00
  - `risk_explanation`: 1.00
- Current bottlenecks:
  - `termination_notice_days`: 0.60
  - `party_b_name`: 0.80
  - `effective_date`: 0.80
  - `term_length`: 0.80

## Risk Engine V2

Implemented in `contractflow/core/risk_engine.py` and the post-extraction orchestration stage in `contractflow/core/extractor.py`, with policy in `docs/risk_policy.json`.

- 3 output classes: `low`, `medium`, `high`
- `risk_level` and `risk_explanation` are derived fields (not directly extracted by the schema prompt)
- weighted factors: liability, governing law region, transfer posture, term, termination, non-solicit
- uncertainty-aware scoring from evidence/confidence coverage
- hard-trigger floors for high-risk combinations
- optional risk-review agent on triggered uncertainty/conflict cases
- optional LLM judge arbitration after deterministic scoring
- normal behavior on uncertainty:
  - missing values remain `unknown`
  - not auto-promoted to `uncapped` or `outside`

Current caveat: current gold risk labels are still single-class (`high`) in this slice. Add balanced low/medium/high risk gold labels for proper calibration.

## Repository Layout

- `contractflow/core/`
  - `pdf_utils.py`: PDF text extraction + OCR fallback
  - `chunking.py`: chunking, BM25/embeddings/hybrid retrieval
  - `extractor.py`: naive/retrieval/field_agents/orchestrated pipelines
  - `risk_engine.py`: policy-driven risk scoring + judge arbitration
- `contractflow/schemas/`
  - `contract_schema.json`
- `contractflow/ui/`
  - `app.py`: FastAPI service for upload, extraction, and risk explainability
  - `templates/index.html`: OpenAI-style minimal UI
  - `static/`: UI CSS + JS
- `scripts/`
  - `baseline_extract.py`, `bulk_extract.py`, `inspect_chunks.py`
  - `run_ui.py`: local web UI launcher
  - `evaluate.py`, `evaluate_risk.py`, `ablation_eval.py`
  - `retrieval_diagnostics.py`, `bootstrap_labels.py`, `build_cuad_pdfs.py`
- `docs/`
  - `domain.md`, `agentic_roadmap.md`, `risk_policy.json`
- `data/`
  - `raw_pdfs/`, `labels/`, `preds_ablations/`, `benchmarks/`

## Quickstart

### 1) Install

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Set API key:

```bash
set OPENAI_API_KEY=your_key_here
```

Optional OCR dependencies (for scanned PDFs): Poppler + Tesseract.

### 2) Run One Document

```bash
# Naive
python scripts/baseline_extract.py data/raw_pdfs/nda_harvard.pdf

# Retrieval context
python scripts/baseline_extract.py data/raw_pdfs/nda_harvard.pdf --retrieval

# Field agents
python scripts/baseline_extract.py data/raw_pdfs/nda_harvard.pdf --field-agents

# Orchestrated with verifier/judge
python scripts/baseline_extract.py data/raw_pdfs/nda_harvard.pdf --orchestrated

# Orchestrated low-cost tuned profile
python scripts/baseline_extract.py data/raw_pdfs/nda_harvard.pdf --orchestrated --orchestrated-profile low_cost

# Orchestrated with risk-review disabled (rules + judge only)
python scripts/baseline_extract.py data/raw_pdfs/nda_harvard.pdf --orchestrated --disable-risk-review

# Override risk-review model and retrieval depth
python scripts/baseline_extract.py data/raw_pdfs/nda_harvard.pdf --orchestrated --risk-review-model gpt-5.2 --risk-review-top-k 5
```

### 2b) Run The Web UI

```bash
python scripts/run_ui.py --host 127.0.0.1 --port 8000 --reload
```

Open `http://127.0.0.1:8000` and:
- upload a PDF
- choose mode (`naive`, `retrieval`, `field_agents`, `orchestrated`)
- choose retrieval backend (`bm25`, `embeddings`, `hybrid`)
- run extraction and inspect:
  - extracted fields
  - explainable risk summary (drivers, protectors, triggers, uncertainty)
  - orchestration trace

If `uvicorn` is missing in your venv, reinstall dependencies:

```bash
pip install -r requirements.txt
```

### 3) Reproduce Evaluation

```bash
# Baseline 3 modes on gold labels (include derived fields)
python scripts/ablation_eval.py --labels-dir data/labels --label-suffix .gold.json --modes naive,retrieval,field_agents --preds-root data/preds_ablations_gold_baseline3_incl --overwrite --include-derived --bootstrap-samples 1000 --out data/benchmarks/gold_ablation_baseline3_include_derived.json

# Tuned orchestrated mode on gold labels
python scripts/ablation_eval.py --labels-dir data/labels --label-suffix .gold.json --modes orchestrated --preds-root data/preds_ablations_gold_orch_tuned_incl --overwrite --orchestrated-profile low_cost --include-derived --bootstrap-samples 1000 --out data/benchmarks/gold_ablation_orchestrated_tuned_include_derived.json

# Optional: evaluate one prediction directory directly
python scripts/evaluate.py --labels-dir data/labels --preds-dir data/preds_ablations_gold_orch_tuned_incl/orchestrated --label-suffix .gold.json --include-derived --bootstrap-samples 1000 --out data/benchmarks/eval_gold_orchestrated_tuned_include_derived.json
```


## Next High-Impact Improvements

1. Build a **gold risk set** with low/medium/high balance.
2. Improve `liability_cap` extraction with a dedicated clause parser and normalization schema.
3. Add **cost-aware orchestration**: dynamic early-exit when verifier confidence is already high.
4. Add **calibration curves** for field confidence and risk confidence.
