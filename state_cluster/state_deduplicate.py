"""Generate annotation-independent groups of exact dataset screenshots."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

from torchvision.io import ImageReadMode, read_image

from state_dataset import StateDataset, read_jsonl


DEDUP_SCOPE_WITHIN_CLUSTER = "within-cluster"
DEDUP_SCOPE_GLOBAL = "global"
DEDUP_SCOPES = (DEDUP_SCOPE_WITHIN_CLUSTER, DEDUP_SCOPE_GLOBAL)
WITHIN_CLUSTER_NOTICE = (
    "Deduplication is applied independently within each current cluster. "
    "Identical screenshots assigned to different clusters will both be retained."
)


def decoded_rgb_sha256(path: Path) -> str:
    """Hash decoded RGB pixels together with their CHW dimensions."""
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


def build_deduplication_groups(
    dataset: StateDataset,
) -> list[dict[str, Any]]:
    """Group every dataset observation by exact decoded screenshot pixels."""
    ids_by_digest: dict[str, list[str]] = defaultdict(list)
    for observation in dataset.observations:
        observation_id = observation["observation_id"]
        digest = decoded_rgb_sha256(dataset.screenshot_path(observation_id))
        ids_by_digest[digest].append(observation_id)
    return [
        {"pixel_sha256": digest, "observation_ids": observation_ids}
        for digest, observation_ids in ids_by_digest.items()
    ]


def validate_deduplication_groups(
    dataset: StateDataset,
    records: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Validate complete, disjoint screenshot groups in dataset order."""
    dataset_order = {
        observation["observation_id"]: index
        for index, observation in enumerate(dataset.observations)
    }
    seen_ids: set[str] = set()
    normalized: list[dict[str, Any]] = []
    previous_first_index = -1
    for record in records:
        if set(record) != {"pixel_sha256", "observation_ids"}:
            raise ValueError("Invalid deduplication record fields")
        digest = record["pixel_sha256"]
        observation_ids = record["observation_ids"]
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError("Invalid pixel_sha256")
        if not isinstance(observation_ids, list) or not observation_ids:
            raise ValueError("Every deduplication group must be non-empty")
        indices: list[int] = []
        for observation_id in observation_ids:
            if observation_id not in dataset_order:
                raise ValueError(
                    f"Unknown deduplicated observation: {observation_id}"
                )
            if observation_id in seen_ids:
                raise ValueError(
                    f"Duplicate deduplicated observation: {observation_id}"
                )
            seen_ids.add(observation_id)
            indices.append(dataset_order[observation_id])
        if indices != sorted(indices):
            raise ValueError(
                "Observation IDs in each deduplication group must follow "
                "dataset order"
            )
        if indices[0] <= previous_first_index:
            raise ValueError(
                "Deduplication groups must follow first occurrence order"
            )
        previous_first_index = indices[0]
        normalized.append(
            {"pixel_sha256": digest, "observation_ids": list(observation_ids)}
        )

    missing_ids = [
        observation["observation_id"]
        for observation in dataset.observations
        if observation["observation_id"] not in seen_ids
    ]
    if missing_ids:
        raise ValueError(
            "Deduplication does not cover observations: " + ", ".join(missing_ids)
        )
    return normalized


def load_deduplication_groups(
    dataset: StateDataset, path: str | Path
) -> list[dict[str, Any]]:
    """Load and validate an annotation-independent deduplication JSONL."""
    return validate_deduplication_groups(dataset, read_jsonl(Path(path)))


def select_representative_ids(
    dataset: StateDataset,
    annotations: Sequence[dict[str, str]],
    groups: Sequence[dict[str, Any]],
    *,
    dedup_scope: str = DEDUP_SCOPE_WITHIN_CLUSTER,
    max_per_image: int = 3,
) -> tuple[str, ...]:
    """Select fixed representative IDs from a source annotation."""
    if dedup_scope not in DEDUP_SCOPES:
        raise ValueError(f"Unknown deduplication scope: {dedup_scope}")
    if max_per_image < 1:
        raise ValueError("max_per_image must be at least 1")

    cluster_by_id = {
        annotation["observation_id"]: annotation["cluster_id"]
        for annotation in annotations
    }
    retained_ids: set[str] = set()
    for group in groups:
        annotated_ids = [
            observation_id
            for observation_id in group["observation_ids"]
            if observation_id in cluster_by_id
        ]
        candidate_groups: list[list[str]]
        if dedup_scope == DEDUP_SCOPE_GLOBAL:
            candidate_groups = [annotated_ids]
        else:
            ids_by_cluster: dict[str, list[str]] = defaultdict(list)
            for observation_id in annotated_ids:
                ids_by_cluster[cluster_by_id[observation_id]].append(
                    observation_id
                )
            candidate_groups = list(ids_by_cluster.values())
        for candidates in candidate_groups:
            retained_ids.update(
                candidates[index]
                for index in evenly_spaced_indices(
                    len(candidates), max_per_image
                )
            )
    return tuple(
        observation["observation_id"]
        for observation in dataset.observations
        if observation["observation_id"] in retained_ids
    )


def _atomic_write_jsonl(
    path: Path, records: Sequence[dict[str, Any]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(path.name + ".tmp")
    try:
        with temporary_path.open("w", encoding="utf-8") as output:
            for record in records:
                output.write(json.dumps(record, ensure_ascii=False) + "\n")
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def create_deduplication_file(
    run_dir: str | Path,
    output_path: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Write canonical exact-screenshot groups without altering the dataset."""
    destination = Path(output_path)
    if destination.exists() and not overwrite:
        raise FileExistsError(
            f"Deduplication file already exists: {destination}"
        )
    dataset = StateDataset.load(run_dir)
    records = build_deduplication_groups(dataset)
    validate_deduplication_groups(dataset, records)
    _atomic_write_jsonl(destination, records)
    return destination


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing deduplication JSONL.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    output = create_deduplication_file(
        args.run_dir, args.output, overwrite=args.overwrite
    )
    print(f"Wrote {len(read_jsonl(output))} screenshot groups to {output}")


if __name__ == "__main__":
    main()
