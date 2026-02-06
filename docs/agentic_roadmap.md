# ContractFlow Agentic Roadmap

This roadmap is optimized for interview signal: clear agent roles, measurable gains, and reproducible evaluation.

## Stage 1: Agent Orchestration Core (Implemented)

- Add `orchestrated` extraction mode with explicit multi-pass workflow:
  - Pass 1: global baseline extraction from retrieved context.
  - Pass 2: per-field extraction agents with evidence and confidence.
  - Pass 3: targeted repair loop for low-confidence/conflicting fields.
- Add candidate selection policy per field using confidence, evidence presence, and validation issues.
- Persist orchestration trace in `_meta.retrieval`:
  - disagreement fields
  - repaired fields
  - selected source counts (`global_baseline`, `field_agent`, `repair_agent`)
  - pass list and baseline coverage

## Stage 2: Retrieval Intelligence (Next)

- Add hybrid retrieval backend (`bm25 + embeddings`) with reciprocal-rank fusion.
- Add field-specific query expansion from schema descriptions and clause aliases.
- Add cross-encoder reranker (optional local model) for top-20 -> top-k reranking.
- Add retrieval diagnostics:
  - MRR / Recall@k against evidence-bearing chunks on labeled subset.
  - per-field retrieval failure report.

## Stage 3: Verifier/Judge Agent (Next)

- Add a dedicated verifier pass that consumes:
  - extracted value
  - evidence snippets
  - deterministic checks (date normalization, term units, risk rules)
- Verifier returns one of: `accept`, `revise`, `unknown`.
- If `revise`, orchestrator triggers retrieval+repair with targeted query templates.
- Track verifier outcomes and disagreement rates in metadata.

## Stage 4: Evaluation Harness (Next)

- Keep current exact/partial metrics and add:
  - bootstrap confidence intervals (already implemented)
  - per-field error buckets (`missing`, `wrong_type`, `wrong_enum`, `span_mismatch`)
  - paired ablation comparison report (`delta`, win/tie/loss per doc)
- Expand gold labels:
  - prioritize 20-30 manually curated gold docs
  - keep silver labels only for broad stress-testing
- Add fixed benchmark command to produce a single JSON report artifact for portfolio.

## Stage 5: Portfolio Packaging (Next)

- Add `demo_run.py` that outputs:
  - final JSON
  - evidence table by field
  - orchestration trace summary
  - risk decision audit
- Add one public benchmark table in README:
  - naive vs retrieval vs field_agents vs orchestrated
  - exact + partial + CI95
- Add failure-case gallery (3-5 docs) with analysis and planned mitigations.
