"""CLI for baseline single-call extraction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from contractflow.core.extractor import ExtractionResult, extract_fields_naive


def main() -> None:
    parser = argparse.ArgumentParser(description="Baseline ContractFlow extractor (single LLM call).")
    parser.add_argument("pdf_path", type=Path, help="Path to the contract PDF")
    parser.add_argument(
        "--schema",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "contractflow" / "schemas" / "contract_schema.json",
        help="Path to the JSON schema describing fields to extract",
    )
    args = parser.parse_args()

    result: ExtractionResult = extract_fields_naive(args.pdf_path, args.schema)
    print(json.dumps(result.json_result, indent=2))


if __name__ == "__main__":
    main()
