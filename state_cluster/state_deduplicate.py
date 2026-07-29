"""Create an observation-only run capped by exact screenshot content."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import struct
import tempfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from torchvision.io import ImageReadMode, read_image

from state_dataset import StateDataset, read_jsonl


def decoded_rgb_sha256(path: Path) -> str:
    """Hash decoded RGB pixels and their CHW dimensions."""
    image = read_image(str(path), mode=ImageReadMode.RGB).contiguous()
    channels, height, width = image.shape
    digest = hashlib.sha256()
    digest.update(struct.pack(">III", channels, height, width))
    digest.update(image.numpy().tobytes(order="C"))
    return digest.hexdigest()


def evenly_spaced_indices(length: int, limit: int) -> tuple[int, ...]:
    """Choose ordered indices including both endpoints, with ties going earlier."""
    if limit < 1:
        raise ValueError("max_per_image must be at least 1")
    if length <= limit:
        return tuple(range(length))
    if limit == 1:
        return (0,)
    return tuple(
        (index * (length - 1)) // (limit - 1) for index in range(limit)
    )


def select_observations(
    dataset: StateDataset, *, max_per_image: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Select observations and describe every repeated exact-image group."""
    if max_per_image < 1:
        raise ValueError("max_per_image must be at least 1")

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for observation in dataset.observations:
        observation_id = observation["observation_id"]
        digest = decoded_rgb_sha256(dataset.screenshot_path(observation_id))
        groups[digest].append(observation)

    retained_ids: set[str] = set()
    audit_records: list[dict[str, Any]] = []
    for digest, observations in groups.items():
        retained = [
            observations[index]
            for index in evenly_spaced_indices(
                len(observations),
                max_per_image,
            )
        ]
        retained_group_ids = [
            observation["observation_id"] for observation in retained
        ]
        retained_ids.update(retained_group_ids)
        if len(observations) > 1:
            original_ids = [
                observation["observation_id"] for observation in observations
            ]
            retained_group_id_set = set(retained_group_ids)
            audit_records.append(
                {
                    "pixel_sha256": digest,
                    "observation_ids": original_ids,
                    "retained_observation_ids": retained_group_ids,
                    "discarded_observation_ids": [
                        observation_id
                        for observation_id in original_ids
                        if observation_id not in retained_group_id_set
                    ],
                }
            )

    selected = [
        observation
        for observation in dataset.observations
        if observation["observation_id"] in retained_ids
    ]
    return selected, audit_records


def select_cluster_assignments(
    dataset: StateDataset,
    assignments: Sequence[dict[str, Any]],
    *,
    max_per_image: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Cap exact-image occurrences independently within each cluster."""
    if max_per_image < 1:
        raise ValueError("max_per_image must be at least 1")

    assignment_by_id: dict[str, dict[str, Any]] = {}
    grouped_ids: dict[str, list[str]] = defaultdict(list)
    for assignment in assignments:
        observation_id = assignment.get("observation_id")
        cluster_id = assignment.get("auto_cluster_id")
        if observation_id not in dataset.observations_by_id:
            raise ValueError(f"Unknown assigned observation: {observation_id}")
        if not isinstance(cluster_id, str) or not cluster_id:
            raise ValueError(
                f"{observation_id} has an invalid auto_cluster_id"
            )
        if observation_id in assignment_by_id:
            raise ValueError(f"Duplicate assignment: {observation_id}")
        assignment_by_id[observation_id] = assignment
        grouped_ids[cluster_id].append(observation_id)

    retained_ids: set[str] = set()
    audit_records: list[dict[str, Any]] = []
    for cluster_id, observation_ids in grouped_ids.items():
        image_groups: dict[str, list[str]] = defaultdict(list)
        for observation_id in observation_ids:
            digest = decoded_rgb_sha256(
                dataset.screenshot_path(observation_id)
            )
            image_groups[digest].append(observation_id)

        for digest, image_observation_ids in image_groups.items():
            retained_group_ids = [
                image_observation_ids[index]
                for index in evenly_spaced_indices(
                    len(image_observation_ids),
                    max_per_image,
                )
            ]
            retained_ids.update(retained_group_ids)
            if len(image_observation_ids) > 1:
                retained_group_id_set = set(retained_group_ids)
                audit_records.append(
                    {
                        "cluster_id": cluster_id,
                        "pixel_sha256": digest,
                        "observation_ids": image_observation_ids,
                        "retained_observation_ids": retained_group_ids,
                        "discarded_observation_ids": [
                            observation_id
                            for observation_id in image_observation_ids
                            if observation_id not in retained_group_id_set
                        ],
                    }
                )

    selected = [
        assignment
        for assignment in assignments
        if assignment["observation_id"] in retained_ids
    ]
    return selected, audit_records


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, records: Sequence[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as output:
        for record in records:
            output.write(json.dumps(record, ensure_ascii=False) + "\n")


def _atomic_write_jsonl(
    path: Path, records: Sequence[dict[str, Any]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(path.name + ".tmp")
    try:
        _write_jsonl(temporary_path, records)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _validate_paths(source: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError(f"Output directory already exists: {output}")
    if output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError(
            "Source and output directories must not contain one another"
        )


def create_deduplicated_run(
    run_dir: str | Path,
    output_dir: str | Path,
    *,
    max_per_image: int = 3,
) -> Path:
    """Create a derived observation-only run without changing the source."""
    if max_per_image < 1:
        raise ValueError("max_per_image must be at least 1")

    source_path = Path(run_dir).resolve()
    output_path = Path(output_dir).resolve()
    _validate_paths(source_path, output_path)
    dataset = StateDataset.load(source_path)
    selected, audit_records = select_observations(
        dataset,
        max_per_image=max_per_image,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = Path(
        tempfile.mkdtemp(
            prefix=f".{output_path.name}.",
            dir=output_path.parent,
        )
    )
    try:
        output_run_id = output_path.name
        output_observations: list[dict[str, Any]] = []
        for observation in selected:
            output_observation = dict(observation)
            output_observation["run_id"] = output_run_id
            output_observations.append(output_observation)
            for field in ("screenshot_path", "view_tree_path"):
                relative_path = Path(observation[field])
                destination = temporary_path / relative_path
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(
                    dataset.resolve_artifact(observation[field]),
                    destination,
                )

        original_count = len(dataset.observations)
        retained_count = len(output_observations)
        manifest = dict(dataset.manifest)
        manifest.update(
            {
                "run_id": output_run_id,
                "status": "derived",
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "observation_count": retained_count,
                "transition_count": 0,
                "failure": None,
                "derivation": {
                    "type": "exact_screenshot_deduplication",
                    "source_run_id": dataset.run_id,
                    "similarity_metric": "sha256_decoded_rgb_pixels",
                    "max_per_image": max_per_image,
                    "original_observation_count": original_count,
                    "retained_observation_count": retained_count,
                    "discarded_observation_count": original_count
                    - retained_count,
                },
            }
        )
        _write_json(temporary_path / "run.json", manifest)
        _write_jsonl(
            temporary_path / "observations.jsonl",
            output_observations,
        )
        (temporary_path / "transitions.jsonl").write_text("", encoding="utf-8")
        _write_jsonl(
            temporary_path / "deduplication.jsonl",
            audit_records,
        )

        StateDataset.load(temporary_path)
        os.replace(temporary_path, output_path)
    except BaseException:
        shutil.rmtree(temporary_path, ignore_errors=True)
        raise
    return output_path


def create_deduplicated_assignments(
    run_dir: str | Path,
    assignments_path: str | Path,
    output_path: str | Path,
    *,
    max_per_image: int = 3,
    audit_path: str | Path | None = None,
) -> tuple[Path, Path]:
    """Write a cluster-preserving subset of assignment records."""
    if max_per_image < 1:
        raise ValueError("max_per_image must be at least 1")

    source_path = Path(assignments_path).resolve()
    destination_path = Path(output_path).resolve()
    if source_path == destination_path:
        raise ValueError("Input and output assignment paths must differ")
    if destination_path.exists():
        raise FileExistsError(
            f"Output assignment file already exists: {destination_path}"
        )
    if not source_path.is_file():
        raise FileNotFoundError(f"Missing assignment file: {source_path}")

    if audit_path is None:
        resolved_audit_path = destination_path.with_name(
            destination_path.stem + "_deduplication.jsonl"
        )
    else:
        resolved_audit_path = Path(audit_path).resolve()
    if resolved_audit_path in (source_path, destination_path):
        raise ValueError(
            "Audit path must differ from input and output assignment paths"
        )
    if resolved_audit_path.exists():
        raise FileExistsError(
            f"Audit file already exists: {resolved_audit_path}"
        )

    dataset = StateDataset.load(run_dir)
    assignments = read_jsonl(source_path)
    selected, audit_records = select_cluster_assignments(
        dataset,
        assignments,
        max_per_image=max_per_image,
    )
    _atomic_write_jsonl(destination_path, selected)
    try:
        _atomic_write_jsonl(resolved_audit_path, audit_records)
    except BaseException:
        destination_path.unlink(missing_ok=True)
        raise
    return destination_path, resolved_audit_path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument(
        "output",
        type=Path,
        help="Derived run directory, or output JSONL with --assignments",
    )
    parser.add_argument(
        "--assignments",
        type=Path,
        help=(
            "Filter this clustering JSONL within each auto_cluster_id instead "
            "of creating a derived run"
        ),
    )
    parser.add_argument(
        "--audit",
        type=Path,
        help="Audit JSONL path for assignment-file mode",
    )
    parser.add_argument("--max-per-image", type=int, default=3)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.assignments is not None:
        output_path, audit_path = create_deduplicated_assignments(
            args.run_dir,
            args.assignments,
            args.output,
            max_per_image=args.max_per_image,
            audit_path=args.audit,
        )
        original_count = len(read_jsonl(args.assignments))
        retained_count = len(read_jsonl(output_path))
        print(
            f"Wrote {output_path} with {retained_count} assignments "
            f"({original_count - retained_count} discarded); "
            f"audit: {audit_path}"
        )
        return
    if args.audit is not None:
        raise ValueError("--audit requires --assignments")
    output_path = create_deduplicated_run(
        args.run_dir,
        args.output,
        max_per_image=args.max_per_image,
    )
    manifest = json.loads(
        (output_path / "run.json").read_text(encoding="utf-8")
    )
    derivation = manifest["derivation"]
    print(
        f"Wrote {output_path} with "
        f"{derivation['retained_observation_count']} observations "
        f"({derivation['discarded_observation_count']} discarded)"
    )


if __name__ == "__main__":
    main()
