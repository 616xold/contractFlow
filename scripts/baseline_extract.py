"""CLI for baseline single-call extraction."""

from __future__ import annotations

import argparse
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

    parser = argparse.ArgumentParser(description="Baseline ContractFlow extractor (single LLM call).")
    parser.add_argument("pdf_path", type=Path, help="Path to the contract PDF")
    parser.add_argument(
        "--schema",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "contractflow" / "schemas" / "contract_schema.json",
        help="Path to the JSON schema describing fields to extract",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Write parsed prediction JSON to this path (default: data/preds/<pdf_stem>.pred.json)",
    )
    parser.add_argument(
        "--raw-out",
        type=Path,
        default=None,
        help="Write raw model output to this path (default: data/preds/<pdf_stem>.raw.txt)",
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

    preds_dir = repo_root / "data" / "preds"
    out_path = args.out or (preds_dir / f"{args.pdf_path.stem}.pred.json")
    raw_out_path = args.raw_out or (preds_dir / f"{args.pdf_path.stem}.raw.txt")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    raw_out_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        result: ExtractionResult = extract_fields_naive(
            args.pdf_path,
            args.schema,
            model=args.model,
            validate=not args.no_validate,
            strict=args.strict,
            coerce=not args.no_coerce,
            structured_outputs=not args.no_structured_outputs,
        )
    except Exception as e:
        print(f"Extraction failed: {e}", file=sys.stderr)
        raise SystemExit(1) from e

    raw_out_path.write_text(result.raw_text, encoding="utf-8")

    pred_payload: dict = dict(result.json_result)
    pred_payload["_meta"] = {
        "pdf": str(args.pdf_path),
        "model": args.model,
        "input_tokens": result.prompt_tokens,
        "output_tokens": result.completion_tokens,
        "validate": not args.no_validate,
        "strict": args.strict,
        "coerce": not args.no_coerce,
        "structured_outputs": not args.no_structured_outputs,
        "issues": result.issues or [],
    }
    out_path.write_text(json.dumps(pred_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(
        f"model={args.model} input_tokens={result.prompt_tokens} output_tokens={result.completion_tokens}",
        file=sys.stderr,
    )
    if result.issues:
        print(f"validation_issues={len(result.issues)}", file=sys.stderr)
    print(f"wrote_pred={out_path}", file=sys.stderr)
    print(f"wrote_raw={raw_out_path}", file=sys.stderr)

    print(json.dumps(result.json_result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
