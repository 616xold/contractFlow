"""Rebuild data/labels/manifest.json from label files on disk."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild label manifest from label files.")
    parser.add_argument(
        "--labels-dir",
        type=Path,
        default=Path("data/labels"),
        help="Directory containing label files",
    )
    parser.add_argument(
        "--silver-mode",
        type=str,
        default="committee",
        help="Mode used for silver labels (default: committee)",
    )
    parser.add_argument(
        "--silver-suffix",
        type=str,
        default=".silver_committee.json",
        help="Suffix used for silver labels (default: .silver_committee.json)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional path to write manifest (default: <labels-dir>/manifest.json)",
    )
    args = parser.parse_args()

    manifest = {}
    now = datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")

    pattern = f"*{args.silver_suffix}"
    for label_path in sorted(args.labels_dir.glob(pattern)):
        doc = label_path.name[: -len(args.silver_suffix)]
        manifest[doc] = {
            "doc": doc,
            "label_file": str(label_path),
            "label_suffix": args.silver_suffix,
            "label_quality": "silver",
            "mode": args.silver_mode,
            "model": "unknown",
            "created_at": now,
        }

    for label_path in sorted(args.labels_dir.glob("*.gold.json")):
        doc = label_path.name[: -len(".gold.json")]
        manifest[doc] = {
            "doc": doc,
            "label_file": str(label_path),
            "label_suffix": ".gold.json",
            "label_quality": "gold",
            "mode": "manual",
            "model": "unknown",
            "created_at": now,
        }

    out_path = args.out or (args.labels_dir / "manifest.json")
    out_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote manifest with {len(manifest)} entries to {out_path}")


if __name__ == "__main__":
    main()
