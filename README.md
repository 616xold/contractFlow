# ContractFlow 🧾

**ContractFlow** is an AI-powered document workflow project focused on **NDAs and simple commercial/SaaS contracts**.

Given one or more PDF contracts, ContractFlow will eventually:

1. Ingest and parse the PDFs  
2. Classify basic document type (e.g. NDA vs “other”)  
3. Extract key legal/business fields into **structured JSON**  
4. Assign a simple **risk score** (low / medium / high) with an explanation  
5. Provide a small UI and workflow integration (e.g. CSV export / Google Sheets / Notion)

The project is designed as a **portfolio‑grade, agentic LLM system** – something you can show to hiring managers at AI/LLM‑heavy companies (e.g. V7, Robin AI, Preprocess, Sprout.ai, consultancies) as evidence that you can build **real LLM/RAG applications around PDFs**, not just chatbots. :contentReference[oaicite:0]{index=0}


## Goals

- Build a **production‑like pipeline** for contract analysis:
  - ingestion → classification → field extraction → risk analysis
- Use **LLMs + retrieval (RAG)** to extract fields reliably from contracts/NDAs
- Show a clear **evaluation story** using labelled gold data and metrics
- Wrap everything in a simple **developer‑friendly interface** (CLI + later a UI)
- Make it easy to **extend** to new contract types or fields


## High‑Level Architecture

> Current status: Week 1 – baseline CLI & plumbing. Later weeks will add the full agent graph, RAG, evaluation and UI. :contentReference[oaicite:1]{index=1}  

Planned components:

- **`ingest/`**
  - PDF → text extraction
  - Chunking, embeddings, vector store (Week 2+)
- **`agents/`**
  - Baseline extractor (single LLM call)
  - Document type classifier (NDA vs other)
  - Field extraction agent with RAG (per‑field retrieval)
  - Risk analysis agent (combines heuristic rules + LLM judgment)
- **`models/`**
  - Pydantic models for structured outputs (contract fields, risk info, full result)
- **`eval/`**
  - Metrics against labelled contracts (per‑field accuracy, overall scores)
  - Simple reporting scripts
- **`ui/`**
  - Streamlit (or lightweight web) app for:
    - uploading PDFs
    - viewing extracted fields + risk
    - exporting/syncing to external tools (CSV / Sheets / Notion)

The orchestration can be implemented with **LangGraph**, a lightweight custom graph of Python functions, or a similar pattern – the key idea is **explicit, composable steps** rather than a single monolithic prompt.


## Tech Stack

- **Language:** Python
- **Backend / Orchestration:** FastAPI (later) + plain Python scripts
- **LLM provider:** Pluggable (e.g. OpenAI / Anthropic / other compatible APIs)
- **PDF/Text:** `pypdf`, `pdfplumber`
- **Data & Storage:**
  - `data/raw_pdfs/` – source documents
  - `data/labels/` – gold‑labelled JSONs for evaluation
  - SQLite/Postgres for metadata (planned)
- **RAG / Vector Store (planned):** Chroma or `pgvector`
- **UI (planned):** Streamlit or small React/Next.js frontend


## Folder Structure

_Current draft structure (will evolve):_

```text
contractflow/
  contractflow/
    __init__.py
    core/
      __init__.py
      pdf_utils.py        # PDF → text utilities
      extractor.py        # Baseline LLM-based extractor
    schemas/
      __init__.py
      contract_schema.json
  data/
    raw_pdfs/             # Sample/real NDAs & contracts (PDF)
    labels/               # Gold label JSONs for eval
  docs/
    domain.md             # Notes on NDAs/contracts & risk rules
  scripts/
    baseline_extract.py   # CLI entrypoint for baseline extractor
  requirements.txt
  README.md

docs/domain.md documents:

what NDAs and SaaS contracts are,

what each field in contract_schema.json means,

simple heuristic rules for risk levels.

Key Fields & Schema

The project focuses on a narrow, fixed schema for NDAs / basic SaaS contracts:

doc_type – high‑level type of document (nda, msa, other)

party_a_name / party_b_name

effective_date

term_length (months/years)

governing_law

termination_notice_days

liability_cap

non_solicit_clause_present (boolean)

data_transfer_outside_uk_eu (boolean/unknown)

risk_level (low / medium / high)

risk_explanation (2–4 sentence natural‑language summary)

The aim is not to cover all legal nuance, but to capture core business‑relevant knobs that lawyers and founders actually care about when skimming contracts. 

ContractFlow

Roadmap (5‑Week Plan)

This repo is intentionally structured around a 5‑week build plan:

Week 1 – Domain, Data & Baseline Pipeline (current)

Decide scope (NDAs + simple SaaS/commercial contracts)

Finalise schema (contract_schema.json) and write docs/domain.md

Collect ~20–30 sample contracts/NDAs (data/raw_pdfs/)

Implement baseline CLI:

pdf_to_text(path) -> str

extract_fields_naive(text) -> dict

For now: single LLM call over full text to fill the schema

Script: python scripts/baseline_extract.py data/raw_pdfs/example.pdf → prints JSON

Deliverable: working baseline CLI on a few PDFs with reasonable JSON output. 

ContractFlow

Week 2 – Agentic Architecture + RAG

Design the agent graph:

IngestionAgent (PDF → chunks → vector store)

DocTypeClassifierAgent (NDA vs other)

FieldExtractionAgent (per‑field retrieval + extraction)

RiskAnalystAgent (risk level + explanation)

RoutingAgent (decide where outputs go)

Implement ingestion, chunking, embeddings, and vector store

Implement classifier & RAG‑based field extraction

Glue into run_pipeline(pdf_path) -> {doc_type, fields, risk}

Week 3 – Evaluation + Annotation

Create a labelled gold set (data/labels/*.gold.json)

Implement evaluation metrics per field and overall

Build a tiny annotation UI (e.g. Streamlit) to review & correct predictions

Use evaluation results to improve prompts, chunking and retrieval

Week 4 – UI, Integrations & Observability

Build a small web UI:

upload documents

view list with doc_type, risk_level, processed_at

view per‑document details (fields + snippets)

Add one integration (CSV export, Google Sheets, or Notion)

Add logging: doc IDs, latency, token usage, risk distribution, eval scores

Week 5 – Polish & Job‑Ready Packaging

Add unit tests for key components

Refactor & document the codebase

Improve README with architecture diagrams & screenshots

Record a short demo video

Write CV/LinkedIn bullets describing ContractFlow as a real project

This staged plan is intentional: it shows iteration and product thinking, not just a one‑off script. 

ContractFlow

Setup
# Clone the repo (once on GitHub)
git clone https://github.com/your-username/contractflow.git
cd contractflow

# Create and activate virtual environment
python -m venv .venv
# Windows:
.\.venv\Scripts\activate
# macOS/Linux:
# source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt


Set your LLM API key in a .env file at the project root (exact variable name depends on the provider, e.g. OPENAI_API_KEY).

Usage (Baseline CLI – Week 1)

Once extract_fields_naive is implemented:

python scripts/baseline_extract.py data/raw_pdfs/nda_001.pdf


This prints a JSON object with:

{
  "doc_type": "nda",
  "fields": {
    "party_a_name": "...",
    "party_b_name": "...",
    "effective_date": "...",
    "term_length": 12,
    "governing_law": "England and Wales",
    "termination_notice_days": 30,
    "liability_cap": "12 months of Fees",
    "non_solicit_clause_present": true,
    "data_transfer_outside_uk_eu": "unknown"
  },
  "risk_level": "medium",
  "risk_explanation": "..."
}


Later weeks will add python scripts/run_pipeline.py for the full agentic + RAG pipeline and a web UI for non‑technical users.

Extending the Project

Some natural extensions once the core is working:

Support additional contract types (e.g. DPAs, MSAs, SOWs)

Add OCR for scanned PDFs

Add more granular risk dimensions (security, data, IP, financial)

Integrate with contract management tools or ticketing systems

Disclaimer

ContractFlow is a learning and portfolio project, not legal advice.
Any outputs should be reviewed by a qualified lawyer before being used in real‑world decision‑making.