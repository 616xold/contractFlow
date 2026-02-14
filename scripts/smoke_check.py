"""One-command local CI smoke check for ContractFlow."""

from __future__ import annotations

import argparse
import py_compile
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Sequence

# Ensure repo root is on PYTHONPATH when running as a script.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from contractflow.core.extractor import DEFAULT_MODEL
from scripts.evaluate_risk_gold import evaluate_risk_gold


def main() -> None:
    parser = argparse.ArgumentParser(description="Run compile + tests + risk-gold eval smoke checks.")
    parser.add_argument(
        "--skip-tests",
        action="store_true",
        help="Skip pytest execution.",
    )
    parser.add_argument(
        "--skip-risk-gold",
        action="store_true",
        help="Skip balanced risk-gold evaluation.",
    )
    parser.add_argument(
        "--risk-gold-dataset",
        type=Path,
        default=REPO_ROOT / "data" / "risk_gold" / "risk_gold_v1.json",
        help="Path to risk-gold dataset JSON.",
    )
    parser.add_argument(
        "--risk-model",
        type=str,
        default=DEFAULT_MODEL,
        help="Model name passed to risk engine during risk-gold checks.",
    )
    parser.add_argument(
        "--min-risk-accuracy",
        type=float,
        default=0.9,
        help="Minimum acceptable risk-gold accuracy threshold (default: 0.9).",
    )
    parser.add_argument(
        "--pytest-args",
        nargs="*",
        default=["tests", "-q"],
        help="Arguments forwarded to pytest (default: tests -q).",
    )
    args = parser.parse_args()

    if not (0.0 <= args.min_risk_accuracy <= 1.0):
        raise ValueError("--min-risk-accuracy must be between 0 and 1.")

    _print_step("Compiling Python sources")
    py_files = list(_iter_python_files((REPO_ROOT / "contractflow", REPO_ROOT / "scripts", REPO_ROOT / "tests")))
    _compile_or_fail(py_files)
    print(f"Compiled {len(py_files)} files.")

    if not args.skip_tests:
        _print_step("Running tests")
        _run_or_fail([sys.executable, "-m", "pytest", *args.pytest_args], cwd=REPO_ROOT)

    if not args.skip_risk_gold:
        _print_step("Evaluating balanced risk-gold set")
        summary = evaluate_risk_gold(
            dataset_path=args.risk_gold_dataset,
            model=args.risk_model,
            enable_judge=False,
            judge_model=None,
            structured_outputs=True,
            use_default_field_meta=True,
            bins=10,
        )
        accuracy = float(summary.get("accuracy", 0.0))
        balanced = bool(summary.get("balanced", False))
        print(
            "risk_gold: "
            f"cases={summary.get('cases_total')} accuracy={accuracy:.4f} "
            f"balance={summary.get('class_balance')}"
        )
        if not balanced:
            raise RuntimeError("Risk-gold dataset is not class-balanced.")
        if accuracy < args.min_risk_accuracy:
            raise RuntimeError(
                f"Risk-gold accuracy {accuracy:.4f} below threshold {args.min_risk_accuracy:.4f}."
            )

    print("\nSmoke check passed.")


def _iter_python_files(roots: Sequence[Path]) -> Iterable[Path]:
    for root in roots:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*.py")):
            if any(part.startswith(".") for part in path.parts):
                continue
            yield path


def _compile_or_fail(py_files: Sequence[Path]) -> None:
    for path in py_files:
        py_compile.compile(str(path), doraise=True)


def _run_or_fail(command: Sequence[str], *, cwd: Path) -> None:
    result = subprocess.run(command, cwd=str(cwd), check=False)
    if result.returncode != 0:
        rendered = " ".join(command)
        raise RuntimeError(f"Command failed ({result.returncode}): {rendered}")


def _print_step(title: str) -> None:
    print(f"\n==> {title}")


if __name__ == "__main__":
    main()
