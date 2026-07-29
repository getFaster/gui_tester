"""Safely migrate legacy cluster assignments into canonical state_data files."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Sequence

from state_dataset import StateDataset, read_jsonl, validate_annotations
from state_deduplicate import create_deduplication_file


def _atomic_write_jsonl(
    path: Path, records: Sequence[dict[str, Any]]
) -> None:
    if path.exists():
        raise FileExistsError(f"Migration output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(path.name + ".tmp")
    try:
        with temporary_path.open("w", encoding="utf-8") as output:
            for record in records:
                output.write(json.dumps(record, ensure_ascii=False) + "\n")
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def convert_legacy_annotation(
    dataset: StateDataset,
    source_path: str | Path,
    output_path: str | Path,
    *,
    source: str | None = None,
) -> Path:
    """Convert legacy automatic IDs while preserving record order and IDs."""
    legacy_records = read_jsonl(Path(source_path))
    baselines = {
        record.get("baseline")
        for record in legacy_records
        if isinstance(record.get("baseline"), str) and record["baseline"]
    }
    invalid_records = [
        record
        for record in legacy_records
        if set(record) != {"observation_id", "auto_cluster_id", "baseline"}
    ]
    if invalid_records:
        raise ValueError(
            "Legacy annotations must contain exactly observation_id, "
            "auto_cluster_id, and baseline"
        )
    if source is None:
        if len(baselines) != 1:
            raise ValueError(
                "Cannot infer source from legacy baseline values; pass source"
            )
        source = next(iter(baselines))

    converted = [
        {
            "observation_id": record.get("observation_id"),
            "cluster_id": record.get("auto_cluster_id"),
            "source": source,
        }
        for record in legacy_records
    ]
    normalized = validate_annotations(
        dataset,
        converted,
        require_complete=True,
    )
    destination = Path(output_path)
    _atomic_write_jsonl(destination, normalized)
    return destination


def migrate_legacy_annotation_in_place(
    dataset: StateDataset,
    annotation_path: str | Path,
    *,
    source: str | None = None,
) -> Path:
    """Replace one legacy file after preserving its exact contents as a backup."""
    path = Path(annotation_path)
    backup_path = path.with_name(f"{path.stem}.legacy{path.suffix}")
    converted_path = path.with_name(f"{path.name}.converted")
    if backup_path.exists():
        raise FileExistsError(f"Migration backup already exists: {backup_path}")
    if converted_path.exists():
        raise FileExistsError(
            f"Temporary migration output already exists: {converted_path}"
        )

    convert_legacy_annotation(
        dataset,
        path,
        converted_path,
        source=source,
    )
    try:
        path.replace(backup_path)
        try:
            converted_path.replace(path)
        except BaseException:
            backup_path.replace(path)
            raise
    finally:
        converted_path.unlink(missing_ok=True)
    return backup_path


def combine_review_logs(
    source_paths: Sequence[str | Path],
    output_path: str | Path,
) -> Path:
    """Combine useful legacy reviews, with later files taking precedence."""
    reviews: dict[str, dict[str, Any]] = {}
    for source_path in source_paths:
        path = Path(source_path)
        if not path.is_file():
            continue
        for record in read_jsonl(path):
            cluster_id = record.get("cluster_id")
            if not isinstance(cluster_id, str) or not cluster_id:
                raise ValueError(f"Invalid review cluster_id in {path}")
            reviews[cluster_id] = record
    destination = Path(output_path)
    _atomic_write_jsonl(destination, list(reviews.values()))
    return destination


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("automatic", type=Path)
    parser.add_argument("corrected", type=Path)
    parser.add_argument("state_data_dir", type=Path)
    parser.add_argument("--automatic-source", default="structure_str")
    parser.add_argument("--corrected-source", default="wikipedia")
    parser.add_argument("--reviews", type=Path, action="append", default=[])
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    dataset = StateDataset.load(args.run_dir)
    annotations_dir = args.state_data_dir / "annotations"
    automatic_output = annotations_dir / f"{args.automatic_source}.jsonl"
    corrected_output = annotations_dir / f"{args.corrected_source}.jsonl"
    deduplication_output = args.state_data_dir / "deduplication.jsonl"
    outputs = (automatic_output, corrected_output, deduplication_output)
    existing = [path for path in outputs if path.exists()]
    if existing:
        raise FileExistsError(
            "Migration output(s) already exist: "
            + ", ".join(str(path) for path in existing)
        )

    convert_legacy_annotation(
        dataset,
        args.automatic,
        automatic_output,
        source=args.automatic_source,
    )
    try:
        convert_legacy_annotation(
            dataset,
            args.corrected,
            corrected_output,
            source=args.corrected_source,
        )
        create_deduplication_file(args.run_dir, deduplication_output)
        if args.reviews:
            combine_review_logs(
                args.reviews,
                args.state_data_dir
                / "debug"
                / f"{args.corrected_source}_reviews.jsonl",
            )
    except BaseException:
        automatic_output.unlink(missing_ok=True)
        corrected_output.unlink(missing_ok=True)
        deduplication_output.unlink(missing_ok=True)
        raise
    print(f"Migrated {dataset.run_id} to {args.state_data_dir}")


if __name__ == "__main__":
    main()
