"""Exhaustively tune element-matching clustering over every effective threshold."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Any, Sequence

import torch

from state_benchmark import pairwise_metrics
from state_clustering import (
    element_matching_distance_matrix,
    load_feature_payloads,
    resolve_similarity_device,
)
from state_dataset import StateDataset, load_annotations
from state_deduplicate import load_deduplication_groups

METRIC_NAMES = (
    "pairwise_f1",
    "adjusted_rand_index",
    "normalized_mutual_information",
)
VIEW_NAMES = ("full", "deduplicated")


def _validate_distance_matrix(distances: torch.Tensor) -> None:
    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("Expected a square distance matrix")
    if distances.shape[0] == 0:
        raise ValueError("At least one observation is required")
    if not torch.isfinite(distances).all():
        raise ValueError("Distance matrix must contain only finite values")
    if not torch.allclose(distances, distances.T, atol=1e-6, rtol=0):
        raise ValueError("Distance matrix must be symmetric")
    if not torch.allclose(
        distances.diagonal(),
        torch.zeros_like(distances.diagonal()),
        atol=1e-6,
        rtol=0,
    ):
        raise ValueError("Distance matrix diagonal must be zero")
    if (distances < 0).any():
        raise ValueError("Distance matrix must not contain negative values")


def _canonical_labels(parent: list[int]) -> list[int]:
    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    cluster_by_root: dict[int, int] = {}
    labels: list[int] = []
    for index in range(len(parent)):
        root = find(index)
        cluster_by_root.setdefault(root, len(cluster_by_root))
        labels.append(cluster_by_root[root])
    return labels


def exhaustive_average_linkage_clusterings(
    distances: torch.Tensor,
) -> list[dict[str, Any]]:
    """Return every distinct average-linkage partition in threshold order."""
    _validate_distance_matrix(distances)
    observation_count = distances.shape[0]
    initial = {
        "distance_threshold": 0.0,
        "merge_distance": None,
        "labels": list(range(observation_count)),
        "predicted_clusters": observation_count,
    }
    if observation_count == 1:
        return [initial]

    from sklearn.cluster import AgglomerativeClustering

    model = AgglomerativeClustering(
        n_clusters=None,
        metric="precomputed",
        linkage="average",
        distance_threshold=0.0,
        compute_full_tree=True,
        compute_distances=True,
    )
    model.fit(distances.cpu().numpy())

    parent = list(range(observation_count))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    representatives = list(range(2 * observation_count - 1))
    clusterings = [initial]
    merge_index = 0
    while merge_index < len(model.distances_):
        merge_distance = float(model.distances_[merge_index])
        group_end = merge_index + 1
        while (
            group_end < len(model.distances_)
            and float(model.distances_[group_end]) == merge_distance
        ):
            group_end += 1

        for current_index in range(merge_index, group_end):
            first_child, second_child = model.children_[current_index]
            first_root = find(representatives[int(first_child)])
            second_root = find(representatives[int(second_child)])
            parent[second_root] = first_root
            representatives[observation_count + current_index] = first_root

        labels = _canonical_labels(parent)
        clusterings.append(
            {
                "distance_threshold": math.nextafter(merge_distance, math.inf),
                "merge_distance": merge_distance,
                "labels": labels,
                "predicted_clusters": len(set(labels)),
            }
        )
        merge_index = group_end
    return clusterings


def metric_values(
    ground_truth: Sequence[str], predicted: Sequence[int]
) -> dict[str, float]:
    """Compute the three threshold-selection metrics."""
    if len(ground_truth) < 2:
        raise ValueError("At least two evaluated observations are required")
    if len(ground_truth) != len(predicted):
        raise ValueError("Ground truth and predictions must have equal length")

    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

    return {
        "pairwise_f1": float(pairwise_metrics(ground_truth, predicted)["pairwise_f1"]),
        "adjusted_rand_index": float(adjusted_rand_score(ground_truth, predicted)),
        "normalized_mutual_information": float(
            normalized_mutual_info_score(ground_truth, predicted)
        ),
    }


def deduplicated_observation_ids(
    groups: Sequence[dict[str, Any]], eligible_ids: set[str]
) -> tuple[str, ...]:
    """Choose the first evaluated observation from each exact-image group."""
    return tuple(
        next(
            observation_id
            for observation_id in group["observation_ids"]
            if observation_id in eligible_ids
        )
        for group in groups
        if any(
            observation_id in eligible_ids
            for observation_id in group["observation_ids"]
        )
    )


def evaluate_clusterings(
    clusterings: Sequence[dict[str, Any]],
    observation_ids: Sequence[str],
    reference: Sequence[dict[str, str]],
    deduplicated_ids: Sequence[str],
) -> list[dict[str, Any]]:
    """Score every clustering on full and exact-image-deduplicated views."""
    reference_by_id = {
        record["observation_id"]: record["cluster_id"] for record in reference
    }
    index_by_id = {
        observation_id: index for index, observation_id in enumerate(observation_ids)
    }
    full_ids = [
        observation_id
        for observation_id in observation_ids
        if observation_id in reference_by_id
    ]
    if len(full_ids) < 2:
        raise ValueError("Reference must cover at least two observations")
    if len(deduplicated_ids) < 2:
        raise ValueError("Deduplicated reference must cover at least two observations")

    ids_by_view = {
        "full": full_ids,
        "deduplicated": list(deduplicated_ids),
    }
    rows: list[dict[str, Any]] = []
    for clustering in clusterings:
        labels = clustering["labels"]
        row = {
            "distance_threshold": clustering["distance_threshold"],
            "merge_distance": clustering["merge_distance"],
            "predicted_clusters": clustering["predicted_clusters"],
        }
        for view_name, view_ids in ids_by_view.items():
            ground_truth = [
                reference_by_id[observation_id] for observation_id in view_ids
            ]
            predicted = [
                labels[index_by_id[observation_id]] for observation_id in view_ids
            ]
            row[f"{view_name}_evaluated_observations"] = len(view_ids)
            row[f"{view_name}_predicted_clusters"] = len(set(predicted))
            for metric_name, value in metric_values(ground_truth, predicted).items():
                row[f"{view_name}_{metric_name}"] = value
        rows.append(row)
    return rows


def best_row(rows: Sequence[dict[str, Any]], *, view_name: str) -> dict[str, Any]:
    """Select by F1, then ARI, NMI, and finally the lower threshold."""
    if view_name not in VIEW_NAMES:
        raise ValueError(f"Unknown evaluation view: {view_name}")
    if not rows:
        raise ValueError("Threshold sweep contains no rows")
    return max(
        rows,
        key=lambda row: (
            row[f"{view_name}_pairwise_f1"],
            row[f"{view_name}_adjusted_rand_index"],
            row[f"{view_name}_normalized_mutual_information"],
            -row["distance_threshold"],
        ),
    )


def row_at_threshold(
    rows: Sequence[dict[str, Any]], threshold: float
) -> dict[str, Any]:
    if not math.isfinite(threshold) or threshold < 0:
        raise ValueError("Default threshold must be finite and non-negative")
    eligible = [row for row in rows if row["distance_threshold"] <= threshold]
    return max(eligible, key=lambda row: row["distance_threshold"])


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(path.name + ".tmp")
    try:
        temporary_path.write_text(content, encoding="utf-8")
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def write_sweep_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("Threshold sweep contains no rows")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(path.name + ".tmp")
    try:
        with temporary_path.open("w", encoding="utf-8", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _polyline_points(
    rows: Sequence[dict[str, Any]],
    field: str,
    *,
    left: float,
    top: float,
    width: float,
    height: float,
    maximum_threshold: float,
    minimum_score: float,
    maximum_score: float,
) -> str:
    points = []
    for row in rows:
        x = left + width * row["distance_threshold"] / maximum_threshold
        score_ratio = (row[field] - minimum_score) / (maximum_score - minimum_score)
        y = top + height * (1 - score_ratio)
        points.append(f"{x:.2f},{y:.2f}")
    return " ".join(points)


def write_metric_plot(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    """Write a dependency-free SVG of full-view metrics over thresholds."""
    if not rows:
        raise ValueError("Threshold sweep contains no rows")
    width, height = 900, 520
    left, top, plot_width, plot_height = 75, 40, 790, 400
    maximum_threshold = max(1e-12, max(row["distance_threshold"] for row in rows))
    minimum_score = min(
        0.0,
        min(row[f"full_{metric_name}"] for row in rows for metric_name in METRIC_NAMES),
    )
    maximum_score = 1.0
    colors = {
        "pairwise_f1": "#2563eb",
        "adjusted_rand_index": "#dc2626",
        "normalized_mutual_information": "#16a34a",
    }
    labels = {
        "pairwise_f1": "Pairwise F1",
        "adjusted_rand_index": "ARI",
        "normalized_mutual_information": "NMI",
    }
    lines = [
        '<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
    ]
    for tick in range(6):
        y = top + plot_height * tick / 5
        value = maximum_score - ((maximum_score - minimum_score) * tick / 5)
        lines.append(
            f'<line x1="{left}" y1="{y:.2f}" '
            f'x2="{left + plot_width}" y2="{y:.2f}" '
            'stroke="#e5e7eb"/>'
        )
        lines.append(
            f'<text x="{left - 12}" y="{y + 4:.2f}" '
            'font-family="sans-serif" font-size="12" text-anchor="end">'
            f"{value:.1f}</text>"
        )
    lines.extend(
        [
            f'<line x1="{left}" y1="{top}" x2="{left}" '
            f'y2="{top + plot_height}" stroke="#111827"/>',
            f'<line x1="{left}" y1="{top + plot_height}" '
            f'x2="{left + plot_width}" y2="{top + plot_height}" '
            'stroke="#111827"/>',
        ]
    )
    for metric_name, color in colors.items():
        points = _polyline_points(
            rows,
            f"full_{metric_name}",
            left=left,
            top=top,
            width=plot_width,
            height=plot_height,
            maximum_threshold=maximum_threshold,
            minimum_score=minimum_score,
            maximum_score=maximum_score,
        )
        lines.append(
            f'<polyline points="{points}" fill="none" stroke="{color}" '
            'stroke-width="2"/>'
        )
    lines.extend(
        [
            f'<text x="{left + plot_width / 2}" y="490" '
            'font-family="sans-serif" font-size="14" text-anchor="middle">'
            "Distance threshold</text>",
            '<text x="18" y="240" font-family="sans-serif" font-size="14" '
            'text-anchor="middle" transform="rotate(-90 18 240)">Score</text>',
            f'<text x="{left}" y="460" font-family="sans-serif" '
            'font-size="12">0</text>',
            f'<text x="{left + plot_width}" y="460" font-family="sans-serif" '
            f'font-size="12" text-anchor="end">{maximum_threshold:.4g}</text>',
        ]
    )
    for index, (metric_name, color) in enumerate(colors.items()):
        legend_x = left + index * 210
        lines.append(
            f'<line x1="{legend_x}" y1="20" x2="{legend_x + 25}" y2="20" '
            f'stroke="{color}" stroke-width="3"/>'
        )
        lines.append(
            f'<text x="{legend_x + 32}" y="24" font-family="sans-serif" '
            f'font-size="13">{labels[metric_name]}</text>'
        )
    lines.append("</svg>")
    _atomic_write_text(path, "\n".join(lines) + "\n")


def _summary_metrics(
    row: dict[str, Any], *, threshold: float | None = None
) -> dict[str, Any]:
    result = {key: value for key, value in row.items() if key != "labels"}
    if threshold is not None:
        result["distance_threshold"] = threshold
    return result


def write_best_assignments(
    path: Path,
    observation_ids: Sequence[str],
    labels: Sequence[int],
) -> None:
    records = [
        {
            "observation_id": observation_id,
            "cluster_id": f"element_matching_{label:04d}",
            "source": "element_matching_threshold_sweep",
        }
        for observation_id, label in zip(observation_ids, labels, strict=True)
    ]
    _atomic_write_text(
        path,
        "".join(json.dumps(record) + "\n" for record in records),
    )


def _json_safe_configuration(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "run_dir": str(args.run_dir.resolve()),
        "feature_dir": str(args.feature_dir.resolve()),
        "reference": str(args.reference.resolve()),
        "deduplication": str(args.deduplication.resolve()),
        "default_threshold": args.default_threshold,
        "clickable_weight": args.clickable_weight,
        "scrollable_weight": args.scrollable_weight,
        "similarity_device": args.similarity_device,
        "tile_chunk_size": args.tile_chunk_size,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("feature_dir", type=Path)
    parser.add_argument("reference", type=Path)
    parser.add_argument("deduplication", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    parser.add_argument("--default-threshold", type=float, default=0.15)
    parser.add_argument("--clickable-weight", type=float, default=1.0)
    parser.add_argument("--scrollable-weight", type=float, default=1.0)
    parser.add_argument("--similarity-device", default="auto")
    parser.add_argument("--tile-chunk-size", type=int, default=512)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    dataset = StateDataset.load(args.run_dir)
    reference = load_annotations(dataset, args.reference)
    deduplication = load_deduplication_groups(dataset, args.deduplication)
    observation_ids = [group["observation_ids"][0] for group in deduplication]
    payloads_by_id = load_feature_payloads(dataset, args.feature_dir, observation_ids)

    distances = element_matching_distance_matrix(
        [payloads_by_id[observation_id] for observation_id in observation_ids],
        class_weights=(args.clickable_weight, args.scrollable_weight),
        device=resolve_similarity_device(args.similarity_device),
        tile_chunk_size=args.tile_chunk_size,
    )
    clusterings = exhaustive_average_linkage_clusterings(distances)
    reference_ids = {record["observation_id"] for record in reference}
    deduplicated_ids = [
        observation_id
        for observation_id in observation_ids
        if observation_id in reference_ids
    ]
    rows = evaluate_clusterings(
        clusterings,
        observation_ids,
        reference,
        deduplicated_ids,
    )
    best_full = best_row(rows, view_name="full")
    best_deduplicated = best_row(rows, view_name="deduplicated")
    default = row_at_threshold(rows, args.default_threshold)
    best_full_clustering = clusterings[rows.index(best_full)]

    output_dir = args.output_dir / dataset.run_id / "element_matching_threshold_sweep"
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(distances, output_dir / "distance_matrix.pt")
    write_sweep_csv(output_dir / "threshold_sweep.csv", rows)
    write_metric_plot(output_dir / "threshold_metrics.svg", rows)
    write_best_assignments(
        output_dir / "best_assignments.jsonl",
        observation_ids,
        best_full_clustering["labels"],
    )
    summary = {
        "run_id": dataset.run_id,
        "selection_rule": (
            "maximum full pairwise F1, then ARI, then NMI, then lower threshold"
        ),
        "effective_thresholds_evaluated": len(rows),
        "configuration": _json_safe_configuration(args),
        "default": _summary_metrics(default, threshold=args.default_threshold),
        "best_full": _summary_metrics(best_full),
        "best_deduplicated": _summary_metrics(best_deduplicated),
    }
    _atomic_write_text(
        output_dir / "summary.json",
        json.dumps(summary, indent=2) + "\n",
    )
    print(
        f"Evaluated {len(rows)} effective thresholds; "
        f"best full-view threshold is "
        f"{best_full['distance_threshold']:.9g} "
        f"(pairwise F1={best_full['full_pairwise_f1']:.6f}, "
        f"ARI={best_full['full_adjusted_rand_index']:.6f}, "
        f"NMI={best_full['full_normalized_mutual_information']:.6f})."
    )
    print(f"Wrote sweep artifacts to {output_dir}")


if __name__ == "__main__":
    main()
