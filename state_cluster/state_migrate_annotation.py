"""Safely convert one legacy annotation JSONL file in place."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from state_dataset import StateDataset
from state_migrate import migrate_legacy_annotation_in_place


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("annotations", type=Path)
    parser.add_argument(
        "--source",
        help="Canonical source label (defaults to the legacy baseline value)",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    dataset = StateDataset.load(args.run_dir)
    backup_path = migrate_legacy_annotation_in_place(
        dataset,
        args.annotations,
        source=args.source,
    )
    print(f"Migrated {args.annotations}")
    print(f"Legacy backup: {backup_path}")


if __name__ == "__main__":
    main()
