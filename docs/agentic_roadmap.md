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

## Stage 2: Retrieval Intelligence (Implemented Core)

- Implemented:
  - hybrid retrieval backend (`bm25 + embeddings`) with reciprocal-rank fusion.
  - field-specific query expansion with clause aliases.
  - optional cross-encoder reranker for top-N -> top-k reranking.
  - retrieval diagnostics script with MRR / Recall@k and per-field failure report.
- Next refinements:
  - add stronger relevance supervision than value-match heuristics.
  - add per-field query templates tuned by contract type.

## Stage 3: Verifier/Judge Agent (Implemented Core)

- Implemented:
  - dedicated verifier pass consuming selected value, evidence, alternative candidates, and deterministic checks.
  - verifier decisions: `accept`, `revise`, `unknown`.
  - judge-triggered retrieval repair loop with query overrides.
  - verifier trace metadata in `_meta.retrieval.orchestration`:
    - decision counts
    - disagreement fields/rate
    - verifier repairs used + fields
- Next refinements:
  - optional second-pass verifier after judge-repair for hard fields.
  - field-specific verifier prompt templates by contract type.

## Risk Engine V2 + Post-Extraction Orchestrator (Implemented Core)

- Implemented:
  - policy-driven risk scoring with versioned config in `docs/risk_policy.json`.
  - structured factor breakdown with contributions, confidence, and evidence coverage.
  - uncertainty-aware scoring adjustments and hard-trigger floors.
  - optional risk judge pass with arbitration (`rules_only`, `rules_with_judge_check`, judge overrides).
  - risk fields (`risk_level`, `risk_explanation`) marked as derived and removed from direct extraction prompts.
  - post-extraction risk orchestration stage:
    - trigger logic from uncertainty + critical-factor confidence.
    - targeted retrieval for risk-input fields.
    - risk-review agent proposes typed field corrections only.
    - deterministic recompute + optional judge on corrected inputs.
  - full risk trace in `_meta.retrieval.risk` for auditability.
- Next refinements:
  - add domain-specific factors (indemnity carve-outs, security obligations, DPA safeguards).
  - calibrate factor weights against manually curated gold risk labels.

## Stage 4: Evaluation Harness (Implemented Core)

- Implemented:
  - exact + partial metrics with bootstrap confidence intervals.
  - per-field and global error buckets (`missing`, `wrong_type`, `wrong_enum`, `span_mismatch`, `value_mismatch`).
  - paired ablation comparison report with per-doc deltas and win/tie/loss counts.
  - fixed benchmark command (`scripts/ablation_eval.py --fixed-benchmark`) that emits one portfolio JSON artifact.
  - committee teacher silver labels as the default silver benchmark set.
- Next refinements:
  - prioritize 20-30 manually curated gold docs for primary reporting.
  - add gold-vs-silver split reporting in one artifact.
  - calibrate confidence bands per field (not only aggregate CI).

## Stage 5: Portfolio Packaging + UI (Implemented Core)

- Implemented:
  - production-style web UI (`contractflow/ui`) with:
    - PDF upload
    - mode selection (`naive`, `retrieval`, `field_agents`, `orchestrated`)
    - retrieval backend selection (`bm25`, `embeddings`, `hybrid`)
    - advanced controls for verifier/risk-review behavior
    - result views for extracted fields, risk summary, orchestration trace, and raw output
  - UI launcher script: `scripts/run_ui.py`
  - README updates for UI and reproducible usage.
- Next refinements:
  - add auth/rate limits for hosted demo deployment.
  - add downloadable audit report per run (JSON + evidence table).
  - add failure-case gallery (3-5 docs) with root-cause analysis + mitigations.
