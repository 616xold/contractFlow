"""Run the ContractFlow web UI."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure repo root is on PYTHONPATH when running as a script.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ContractFlow UI server.")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host bind address")
    parser.add_argument("--port", type=int, default=8000, help="Port number")
    parser.add_argument("--reload", action="store_true", help="Enable code reload (dev mode)")
    args = parser.parse_args()

    from dotenv import load_dotenv
    from uvicorn import run as uvicorn_run

    load_dotenv(REPO_ROOT / ".env")
    uvicorn_run("contractflow.ui.app:app", host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()

