ContractFlow Overview
=====================

ContractFlow is a baseline contract extractor focused on NDAs and simple commercial agreements.
It reads PDFs, extracts a fixed schema into JSON, and computes a risk summary. The project
is structured to evolve into an agentic pipeline with retrieval, evidence, validation, and
evaluation.

Goals
-----
- Extract structured fields from contracts (party names, dates, term, governing law, etc).
- Provide deterministic validation and normalization of outputs.
- Support agentic retrieval and per-field extraction with evidence.
- Track evaluation metrics against gold labels.

Repository Layout
-----------------
- contractflow/core/
  - pdf_utils.py: PDF text extraction (per-page + full text), optional OCR fallback.
  - chunking.py: Chunking, BM25 retrieval, embeddings retrieval, and retrieval helpers.
  - extractor.py: LLM extraction pipelines (naive, retrieval context, field agents).
- contractflow/schemas/
  - contract_schema.json: JSON schema for all extraction fields.
  - models.py: Pydantic model for structured outputs.
- scripts/
  - baseline_extract.py: Single-document extractor CLI.
  - bulk_extract.py: Batch extractor CLI over a folder of PDFs.
  - inspect_chunks.py: Prints chunk headings and snippets for tuning chunking.
  - evaluate.py: Compares predictions vs gold labels and reports accuracy + coverage.
  - ablation_eval.py: Runs naive vs retrieval vs field_agents and evaluates each.
  - retrieval_diagnostics.py: Reports retrieval MRR/Recall@k and per-field failures.
  - build_cuad_pdfs.py: Generates PDFs from the public CUAD dataset text.
  - bootstrap_labels.py: Generates silver labels from an extraction mode.
- data/
  - raw_pdfs/: source documents.
  - preds/: extraction outputs and raw model outputs.
  - labels/: gold and silver labels, labeling templates, and manifest.
- docs/
  - domain.md: field definitions and risk heuristics.
  - agentic_roadmap.md: staged plan for agentic orchestration and evaluation upgrades.

Data Flow (High-Level)
----------------------
1) PDF ingestion
   - pdf_utils.read_pdf_pages() extracts per-page text using pypdf.
2) Chunking + retrieval (optional)
   - chunking.chunk_pdf() splits pages into sections using heading heuristics.
   - Retrieval uses BM25 or embeddings and returns top-k chunk hits.
3) LLM extraction
   - Naive: one call over the full document.
   - Retrieval context: one call over concatenated retrieved excerpts.
   - Field agents: per-field calls using retrieved excerpts, evidence snippets, and confidence.
4) Verification and normalization
   - Type validation/coercion per schema.
   - Deterministic normalization for effective_date and term_length.
   - Risk level and explanation derived from domain heuristics.
5) Evaluation
   - evaluate.py compares predictions to gold labels and reports accuracy and coverage.

Extraction Modes
----------------
1) Naive (single call)
   - extractor.extract_fields_naive(): full PDF text -> one LLM call -> schema validation.

2) Retrieval context (single call with excerpts)
   - extractor.extract_fields_retrieval():
     - Build chunk index.
     - Retrieve top-k chunks per field.
     - Assemble excerpts into a single prompt.
     - One LLM call with schema validation.
     - _meta.retrieval.coverage reports retrieval hit coverage.

3) Field agents (per-field agent loop)
   - extractor.extract_fields_field_agents():
     - Build chunk index.
     - For each field:
       - Build a field-specific query.
       - Retrieve top-k chunks.
       - Run a per-field LLM call that returns value + evidence + confidence.
       - Retry with augmented query when confidence is low or conflicts detected.
     - Apply deterministic verifiers and risk heuristics.
     - _meta.retrieval.coverage reports evidence coverage.

4) Orchestrated agents (multi-pass)
   - extractor.extract_fields_orchestrated():
     - Global baseline extraction from retrieved context.
     - Per-field agent extraction with evidence/confidence.
     - Candidate selection and disagreement detection per field.
     - Targeted repair loop for low-confidence/conflicting fields.
     - Verifier/judge pass returns `accept` / `revise` / `unknown`.
     - Judge-triggered repair loop for uncertain/conflicting fields.
     - _meta.retrieval.orchestration records pass-level trace, verifier outcomes, and disagreement rate.

Chunking and Retrieval
----------------------
- chunking.py detects headings using:
  - Section/Article prefixes.
  - Numbered headings (e.g. "1. Definitions").
  - All-caps lines.
  - Title-case headings.
- Retrieval backends:
  - BM25 (local scoring).
  - Embeddings via OpenAI embeddings API with cosine similarity.
  - Hybrid BM25 + embeddings with reciprocal-rank fusion (RRF).
- Optional reranker:
  - Cross-encoder reranking over top-N candidates (requires sentence-transformers).
- Both return top-k chunks with page numbers and headings.

Evidence and Coverage
---------------------
- Field agents return evidence snippets and a confidence score.
- Coverage metrics:
  - Retrieval context: hit_ratio based on retrieval hits per field.
  - Field agents: evidence_ratio and confidence stats based on evidence snippets.

Schema and Validation
---------------------
- contract_schema.json is the source of truth for field definitions and types.
- extractor.py enforces:
  - Required keys and types.
  - Enum constraints.
  - Non-empty strings for required fields.
  - "unknown" reserved for data_transfer_outside_uk_eu only.
- Normalization:
  - effective_date normalized to ISO if possible.
  - term_length normalized to months if specified in years.

Risk Heuristics
---------------
Risk is derived deterministically from docs/domain.md:
- Start from medium.
- High if liability is uncapped, governing law is outside UK/EU, or data transfer is "yes".
- Low if liability <= 12 months, governing law is England & Wales, and term <= 12 months.

CLI Usage Examples
------------------
- Single extraction (naive):
  python scripts/baseline_extract.py data/raw_pdfs/nda_harvard.pdf

- Retrieval context (BM25):
  python scripts/baseline_extract.py data/raw_pdfs/nda_harvard.pdf --retrieval

- Field agents (BM25):
  python scripts/baseline_extract.py data/raw_pdfs/nda_harvard.pdf --field-agents

- Orchestrated multi-pass extraction:
  python scripts/baseline_extract.py data/raw_pdfs/nda_harvard.pdf --orchestrated

- Orchestrated extraction without verifier:
  python scripts/baseline_extract.py data/raw_pdfs/nda_harvard.pdf --orchestrated --disable-verifier

- Orchestrated extraction with custom verifier settings:
  python scripts/baseline_extract.py data/raw_pdfs/nda_harvard.pdf --orchestrated --verifier-confidence-threshold 0.65 --verifier-max-repairs 6

- Field agents (embeddings backend):
  python scripts/baseline_extract.py data/raw_pdfs/nda_harvard.pdf --field-agents --retrieval-backend embeddings

- Field agents (hybrid backend):
  python scripts/baseline_extract.py data/raw_pdfs/nda_harvard.pdf --field-agents --retrieval-backend hybrid

- Field agents with reranker:
  python scripts/baseline_extract.py data/raw_pdfs/nda_harvard.pdf --field-agents --retrieval-backend hybrid --reranker-model cross-encoder/ms-marco-MiniLM-L-6-v2 --reranker-top-n 20

- Inspect chunk headings:
  python scripts/inspect_chunks.py data/raw_pdfs/nda_harvard.pdf --max-chars 200

- Evaluate predictions:
  python scripts/evaluate.py --labels-dir data/labels --preds-dir data/preds

- Evaluate with bootstrap confidence intervals:
  python scripts/evaluate.py --labels-dir data/labels --preds-dir data/preds --bootstrap-samples 1000

- Evaluate against silver labels:
  python scripts/evaluate.py --labels-dir data/labels --label-suffix .silver.json --preds-dir data/preds

- Run ablations (naive vs retrieval vs field_agents vs orchestrated):
  python scripts/ablation_eval.py --labels-dir data/labels --label-suffix .silver.json

- Retrieval diagnostics (MRR/Recall@k + failure report):
  python scripts/retrieval_diagnostics.py --labels-dir data/labels --label-suffix .silver.json --retrieval-backend hybrid --top-k 5 --k-values 1,3,5 --out data/retrieval_diag_hybrid.json

- Generate PDFs from CUAD text:
  python scripts/build_cuad_pdfs.py --limit 25

- Bootstrap silver labels:
  python scripts/bootstrap_labels.py --label-suffix .silver.json --mode field_agents

Known Gaps and Next Steps
-------------------------
- Expand labeled datasets in data/labels/.
- Improve heading heuristics and chunking for long contracts.
- OCR fallback requires system dependencies (Poppler for pdf2image, Tesseract for pytesseract).
- Curate more gold labels (silver labels are not a reliable benchmark).
- Add stronger retrieval relevance labels beyond value-match heuristics.
- Optional reranker dependency: `pip install sentence-transformers`.
