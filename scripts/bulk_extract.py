"""Bulk runner for baseline extraction over a directory of PDFs."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

# Ensure repo root is on PYTHONPATH when running as a script.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from contractflow.core.extractor import DEFAULT_MODEL, ExtractionResult, extract_fields_naive


def main() -> None:
    repo_root = REPO_ROOT

    from dotenv import load_dotenv

    load_dotenv(repo_root / ".env")

    parser = argparse.ArgumentParser(description="Run baseline extraction over all PDFs in a folder.")
    parser.add_argument(
        "--in-dir",
        type=Path,
        default=repo_root / "data" / "raw_pdfs",
        help="Directory containing input PDFs",
    )
    parser.add_argument(
        "--preds-dir",
        type=Path,
        default=repo_root / "data" / "preds",
        help="Directory to write prediction outputs",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Path to write a CSV summary (default: <preds-dir>/summary.csv)",
    )
    parser.add_argument(
        "--schema",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "contractflow" / "schemas" / "contract_schema.json",
        help="Path to the JSON schema describing fields to extract",
    )
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="OpenAI model name")
    parser.add_argument("--no-validate", action="store_true", help="Disable schema validation (debugging)")
    parser.add_argument("--strict", action="store_true", help="Fail fast on schema issues (default: lenient)")
    parser.add_argument("--no-coerce", action="store_true", help="Disable type coercion (debugging)")
    parser.add_argument(
        "--no-structured-outputs",
        action="store_true",
        help="Disable structured outputs parsing (debugging / fallback mode)",
    )
    args = parser.parse_args()

    pdf_paths = sorted(args.in_dir.glob("*.pdf"))
    if not pdf_paths:
        print(f"No PDFs found in {args.in_dir}", file=sys.stderr)
        raise SystemExit(1)

    args.preds_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.csv or (args.preds_dir / "summary.csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "pdf",
        "model",
        "input_tokens",
        "output_tokens",
        "success",
        "error",
    ]

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for pdf_path in pdf_paths:
            out_path = args.preds_dir / f"{pdf_path.stem}.pred.json"
            raw_out_path = args.preds_dir / f"{pdf_path.stem}.raw.txt"

            try:
                result: ExtractionResult = extract_fields_naive(
                    pdf_path,
                    args.schema,
                    model=args.model,
                    validate=not args.no_validate,
                    strict=args.strict,
                    coerce=not args.no_coerce,
                    structured_outputs=not args.no_structured_outputs,
                )
                raw_out_path.write_text(result.raw_text, encoding="utf-8")

                pred_payload: dict = dict(result.json_result)
                pred_payload["_meta"] = {
                    "pdf": str(pdf_path),
                    "model": args.model,
                    "input_tokens": result.prompt_tokens,
                    "output_tokens": result.completion_tokens,
                    "validate": not args.no_validate,
                    "strict": args.strict,
                    "coerce": not args.no_coerce,
                    "structured_outputs": not args.no_structured_outputs,
                    "issues": result.issues or [],
                }
                out_path.write_text(
                    json.dumps(pred_payload, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8",
                )
                writer.writerow(
                    {
                        "pdf": pdf_path.name,
                        "model": args.model,
                        "input_tokens": result.prompt_tokens,
                        "output_tokens": result.completion_tokens,
                        "success": 1,
                        "error": "",
                    }
                )
                if result.issues:
                    print(f"WARN {pdf_path.name} issues={len(result.issues)}", file=sys.stderr)
                else:
                    print(f"OK   {pdf_path.name}", file=sys.stderr)
            except Exception as e:
                error = str(e).replace("\r\n", "\\n").replace("\n", "\\n")
                writer.writerow(
                    {
                        "pdf": pdf_path.name,
                        "model": args.model,
                        "input_tokens": "",
                        "output_tokens": "",
                        "success": 0,
                        "error": error,
                    }
                )
                print(f"FAIL {pdf_path.name}: {e}", file=sys.stderr)

    print(f"Wrote summary CSV to {csv_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
