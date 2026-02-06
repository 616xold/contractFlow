"""Retrieval diagnostics: MRR/Recall@k and per-field failure analysis.

This script uses a value-match relevance heuristic: a chunk is relevant for a field
if it contains the normalized gold value for that field.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Optional

# Ensure repo root is on PYTHONPATH when running as a script.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from contractflow.core.chunking import RetrievalHit, build_retriever, chunk_pdf
from contractflow.core.extractor import build_field_queries, load_schema


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate retrieval quality with MRR/Recall@k.")
    parser.add_argument(
        "--in-dir",
        type=Path,
        default=REPO_ROOT / "data" / "raw_pdfs",
        help="Directory containing input PDFs",
    )
    parser.add_argument(
        "--labels-dir",
        type=Path,
        default=REPO_ROOT / "data" / "labels",
        help="Directory containing label files",
    )
    parser.add_argument(
        "--label-suffix",
        type=str,
        default=".gold.json",
        help="Label filename suffix (default: .gold.json)",
    )
    parser.add_argument(
        "--schema",
        type=Path,
        default=REPO_ROOT / "contractflow" / "schemas" / "contract_schema.json",
        help="Path to schema JSON",
    )
    parser.add_argument(
        "--retrieval-backend",
        type=str,
        default="bm25",
        help="Retrieval backend: bm25, embeddings, or hybrid (default: bm25)",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="text-embedding-3-small",
        help="Embedding model for embeddings/hybrid backend",
    )
    parser.add_argument(
        "--embedding-batch-size",
        type=int,
        default=64,
        help="Embedding batch size (default: 64)",
    )
    parser.add_argument(
        "--embedding-cache-dir",
        type=Path,
        default=REPO_ROOT / "data" / "embeddings",
        help="Directory for embedding cache files",
    )
    parser.add_argument(
        "--reranker-model",
        type=str,
        default=None,
        help="Optional cross-encoder reranker model (requires sentence-transformers)",
    )
    parser.add_argument(
        "--reranker-top-n",
        type=int,
        default=20,
        help="Candidate pool size before reranking (default: 20)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Top-k results to evaluate (default: 5)",
    )
    parser.add_argument(
        "--k-values",
        type=str,
        default="1,3,5",
        help="Comma-separated recall cutoffs (default: 1,3,5)",
    )
    parser.add_argument(
        "--chunk-max-chars",
        type=int,
        default=2000,
        help="Max chars per chunk during chunking (default: 2000)",
    )
    parser.add_argument(
        "--exclude-fields",
        type=str,
        default="risk_explanation",
        help="Comma-separated fields to exclude from diagnostics",
    )
    parser.add_argument(
        "--max-docs",
        type=int,
        default=None,
        help="Optional max number of docs to process",
    )
    parser.add_argument(
        "--failures-limit",
        type=int,
        default=200,
        help="Maximum failed examples to keep in output (default: 200)",
    )
    parser.add_argument(
        "--use-ocr",
        action="store_true",
        help="Enable OCR fallback when extracted text is sparse",
    )
    parser.add_argument(
        "--ocr-min-chars",
        type=int,
        default=40,
        help="Min avg chars per page before OCR fallback",
    )
    parser.add_argument(
        "--ocr-lang",
        type=str,
        default="eng",
        help="OCR language (default: eng)",
    )
    parser.add_argument(
        "--ocr-dpi",
        type=int,
        default=200,
        help="OCR DPI for pdf2image (default: 200)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional path to write JSON report",
    )
    args = parser.parse_args()

    from dotenv import load_dotenv

    load_dotenv(REPO_ROOT / ".env")

    k_values = _parse_k_values(args.k_values, args.top_k)
    excluded_fields = {
        field.strip()
        for field in args.exclude_fields.split(",")
        if field.strip()
    }

    schema = load_schema(args.schema)
    queries = build_field_queries(schema)
    label_paths = sorted(args.labels_dir.glob(f"*{args.label_suffix}"))
    if args.max_docs is not None:
        label_paths = label_paths[: max(0, args.max_docs)]
    if not label_paths:
        raise ValueError(f"No labels found in {args.labels_dir} with suffix {args.label_suffix}")

    per_field: Dict[str, Dict[str, Any]] = {}
    for field in schema.keys():
        if field in excluded_fields:
            continue
        per_field[field] = {
            "opportunities": 0,
            "mrr_sum": 0.0,
            "hits_at_k": {str(k): 0 for k in k_values},
            "missing_label": 0,
            "no_relevant_chunks": 0,
            "retrieval_failures": 0,
        }

    docs_total = len(label_paths)
    docs_evaluated = 0
    docs_missing_pdf = 0
    fields_opportunities = 0
    overall_mrr_sum = 0.0
    overall_hits_at_k = {str(k): 0 for k in k_values}
    failures: list[Dict[str, Any]] = []

    for label_path in label_paths:
        doc_id = _base_name_from_label(label_path.name, args.label_suffix)
        pdf_path = args.in_dir / f"{doc_id}.pdf"
        if not pdf_path.exists():
            docs_missing_pdf += 1
            continue
        gold = _load_json(label_path)

        chunks = chunk_pdf(
            pdf_path,
            max_chunk_chars=args.chunk_max_chars,
            use_ocr=args.use_ocr,
            ocr_min_chars=args.ocr_min_chars,
            ocr_lang=args.ocr_lang,
            ocr_dpi=args.ocr_dpi,
        )
        if not chunks:
            continue
        docs_evaluated += 1

        retriever = build_retriever(
            chunks,
            backend=args.retrieval_backend,
            embedding_model=args.embedding_model,
            embedding_batch_size=args.embedding_batch_size,
            embedding_cache_dir=args.embedding_cache_dir,
            reranker_model=args.reranker_model,
            reranker_top_n=args.reranker_top_n,
        )

        for field, meta in schema.items():
            if field in excluded_fields:
                continue
            field_stats = per_field[field]
            gold_value = gold.get(field)
            targets = _value_targets(gold_value, meta)
            if not targets:
                field_stats["missing_label"] += 1
                continue

            relevant_chunks = [
                chunk.chunk_id
                for chunk in chunks
                if _chunk_contains_any_target(chunk.combined_text(), targets)
            ]
            if not relevant_chunks:
                field_stats["no_relevant_chunks"] += 1
                continue
            relevant_ids = set(relevant_chunks)

            query = queries.get(field, field.replace("_", " "))
            hits = retriever.retrieve(query, top_k=args.top_k)
            rank = _first_relevant_rank(hits, relevant_ids)

            fields_opportunities += 1
            field_stats["opportunities"] += 1
            if rank is not None:
                rr = 1.0 / rank
                overall_mrr_sum += rr
                field_stats["mrr_sum"] += rr
                for k in k_values:
                    if rank <= k:
                        overall_hits_at_k[str(k)] += 1
                        field_stats["hits_at_k"][str(k)] += 1
            else:
                field_stats["retrieval_failures"] += 1
                if len(failures) < args.failures_limit:
                    failures.append(
                        {
                            "doc": doc_id,
                            "field": field,
                            "query": query,
                            "gold_value": gold_value,
                            "targets": targets[:5],
                            "top_hits": [
                                {
                                    "chunk_id": hit.chunk.chunk_id,
                                    "page_num": hit.chunk.page_num,
                                    "heading": hit.chunk.heading,
                                    "score": round(hit.score, 4),
                                }
                                for hit in hits[:3]
                            ],
                        }
                    )

    overall = {
        "opportunities": fields_opportunities,
        "mrr": round((overall_mrr_sum / fields_opportunities) if fields_opportunities else 0.0, 4),
    }
    for k in k_values:
        hit_count = overall_hits_at_k[str(k)]
        overall[f"recall@{k}"] = round((hit_count / fields_opportunities) if fields_opportunities else 0.0, 4)

    per_field_report: Dict[str, Any] = {}
    for field, stats in per_field.items():
        opportunities = stats["opportunities"]
        field_report = {
            "opportunities": opportunities,
            "mrr": round((stats["mrr_sum"] / opportunities) if opportunities else 0.0, 4),
            "missing_label": stats["missing_label"],
            "no_relevant_chunks": stats["no_relevant_chunks"],
            "retrieval_failures": stats["retrieval_failures"],
        }
        for k in k_values:
            hits = stats["hits_at_k"][str(k)]
            field_report[f"recall@{k}"] = round((hits / opportunities) if opportunities else 0.0, 4)
        per_field_report[field] = field_report

    report = {
        "config": {
            "retrieval_backend": args.retrieval_backend,
            "embedding_model": args.embedding_model,
            "embedding_batch_size": args.embedding_batch_size,
            "embedding_cache_dir": str(args.embedding_cache_dir),
            "reranker_model": args.reranker_model,
            "reranker_top_n": args.reranker_top_n if args.reranker_model else None,
            "top_k": args.top_k,
            "k_values": k_values,
            "chunk_max_chars": args.chunk_max_chars,
            "label_suffix": args.label_suffix,
            "excluded_fields": sorted(excluded_fields),
            "use_ocr": args.use_ocr,
        },
        "docs_total": docs_total,
        "docs_evaluated": docs_evaluated,
        "docs_missing_pdf": docs_missing_pdf,
        "overall": overall,
        "per_field": per_field_report,
        "failures": failures,
    }

    _print_report(report, k_values)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _base_name_from_label(filename: str, label_suffix: str) -> str:
    if filename.endswith(label_suffix):
        return filename[: -len(label_suffix)]
    return Path(filename).stem


def _parse_k_values(raw: str, top_k: int) -> list[int]:
    parsed = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        if not token.isdigit():
            raise ValueError(f"Invalid k value: {token!r}")
        value = int(token)
        if value < 1:
            raise ValueError(f"k values must be >= 1, got {value}")
        if value <= top_k:
            parsed.append(value)
    if top_k not in parsed:
        parsed.append(top_k)
    return sorted(set(parsed))


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return data


def _value_targets(value: Any, meta: Dict[str, Any]) -> list[str]:
    if value is None:
        return []
    expected_type = meta.get("type")
    if expected_type == "boolean":
        if value is True:
            return ["true", "yes"]
        if value is False:
            return ["false", "no"]
        return []
    if expected_type == "integer":
        if isinstance(value, int):
            return [str(value)]
        if isinstance(value, str):
            match = re.search(r"-?\d+", value.replace(",", ""))
            return [match.group(0)] if match else []
        return []

    text = str(value).strip()
    if not text:
        return []

    normalized = _normalize_text(text)
    if not normalized:
        return []

    targets = [normalized]
    number_match = re.search(r"-?\d+", normalized)
    if number_match:
        targets.append(number_match.group(0))
    return list(dict.fromkeys(targets))


def _chunk_contains_any_target(text: str, targets: list[str]) -> bool:
    haystack = _normalize_text(text)
    if not haystack:
        return False
    for target in targets:
        if not target:
            continue
        if target.isdigit() or (target.startswith("-") and target[1:].isdigit()):
            if re.search(rf"(?<!\d){re.escape(target)}(?!\d)", haystack):
                return True
        elif target in haystack:
            return True
    return False


def _first_relevant_rank(hits: list[RetrievalHit], relevant_ids: set[str]) -> Optional[int]:
    for idx, hit in enumerate(hits, start=1):
        if hit.chunk.chunk_id in relevant_ids:
            return idx
    return None


def _normalize_text(text: str) -> str:
    lowered = text.lower()
    lowered = re.sub(r"[^a-z0-9]+", " ", lowered)
    return " ".join(lowered.split())


def _print_report(report: Dict[str, Any], k_values: list[int]) -> None:
    print(
        f"docs_total={report['docs_total']} docs_evaluated={report['docs_evaluated']} "
        f"docs_missing_pdf={report['docs_missing_pdf']}"
    )
    overall = report["overall"]
    print(f"overall_mrr={overall['mrr']} opportunities={overall['opportunities']}")
    for k in k_values:
        print(f"overall_recall@{k}={overall.get(f'recall@{k}')}")
    print("per_field:")
    for field, stats in report["per_field"].items():
        pieces = [f"mrr={stats['mrr']}", f"opps={stats['opportunities']}"]
        for k in k_values:
            pieces.append(f"r@{k}={stats.get(f'recall@{k}')}")
        pieces.append(f"failures={stats['retrieval_failures']}")
        print(f"  {field}: " + " ".join(pieces))
    print(f"failure_examples={len(report['failures'])}")


if __name__ == "__main__":
    main()
