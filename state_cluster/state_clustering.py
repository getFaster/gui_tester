"""Build automatic state-clustering baselines for one collected run."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn.functional as F

from state_dataset import StateDataset, read_jsonl
from state_features import hamming_distance


ELEMENT_CLASS_NAMES = ("clickable", "scrollable")


def categorical_clusters(
    observations: Sequence[dict[str, Any]], field: str
) -> list[int]:
    """Assign the same integer to equal categorical values."""
    labels_by_value: dict[Any, int] = {}
    labels: list[int] = []
    for observation in observations:
        value = observation.get(field)
        if value not in labels_by_value:
            labels_by_value[value] = len(labels_by_value)
        labels.append(labels_by_value[value])
    return labels


def perceptual_hash_clusters(
    hashes: Sequence[str], *, max_hamming_distance: int = 8
) -> list[int]:
    """Cluster hashes by connected components under a Hamming threshold."""
    parent = list(range(len(hashes)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(first: int, second: int) -> None:
        first_root = find(first)
        second_root = find(second)
        if first_root != second_root:
            parent[second_root] = first_root

    for first_index, first_hash in enumerate(hashes):
        for second_index in range(first_index):
            if (
                hamming_distance(first_hash, hashes[second_index])
                <= max_hamming_distance
            ):
                union(first_index, second_index)

    cluster_by_root: dict[int, int] = {}
    labels: list[int] = []
    for index in range(len(hashes)):
        root = find(index)
        cluster_by_root.setdefault(root, len(cluster_by_root))
        labels.append(cluster_by_root[root])
    return labels


def embedding_clusters(
    embeddings: torch.Tensor, *, distance_threshold: float
) -> list[int]:
    """Agglomeratively cluster normalized embeddings with cosine distance."""
    if embeddings.ndim != 2:
        raise ValueError(
            f"Expected (observations, features), got {tuple(embeddings.shape)}"
        )
    if embeddings.shape[0] == 0:
        return []
    if embeddings.shape[0] == 1:
        return [0]
    from sklearn.cluster import AgglomerativeClustering

    normalized = F.normalize(embeddings.to(dtype=torch.float32), dim=1)
    model = AgglomerativeClustering(
        n_clusters=None,
        metric="cosine",
        linkage="average",
        distance_threshold=distance_threshold,
    )
    return model.fit_predict(normalized.cpu().numpy()).tolist()


def distance_matrix_clusters(
    distances: torch.Tensor, *, distance_threshold: float
) -> list[int]:
    """Agglomeratively cluster a symmetric precomputed distance matrix."""
    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError(
            "Expected a square (observations, observations) distance matrix, "
            f"got {tuple(distances.shape)}"
        )
    if not torch.isfinite(distances).all():
        raise ValueError("Distance matrix must contain only finite values")
    if not torch.allclose(distances, distances.T, atol=1e-6, rtol=0):
        raise ValueError("Distance matrix must be symmetric")
    if not torch.allclose(
        distances.diagonal(),
        torch.zeros(
            distances.shape[0],
            dtype=distances.dtype,
            device=distances.device,
        ),
        atol=1e-6,
        rtol=0,
    ):
        raise ValueError("Distance matrix diagonal must be zero")
    if (distances < 0).any():
        raise ValueError("Distance matrix must not contain negative values")
    if distances.shape[0] == 0:
        return []
    if distances.shape[0] == 1:
        return [0]

    from sklearn.cluster import AgglomerativeClustering

    model = AgglomerativeClustering(
        n_clusters=None,
        metric="precomputed",
        linkage="average",
        distance_threshold=distance_threshold,
    )
    return model.fit_predict(distances.cpu().numpy()).tolist()


def load_feature_payloads(
    dataset: StateDataset, feature_dir: Path
) -> dict[str, dict[str, Any]]:
    payloads: dict[str, dict[str, Any]] = {}
    for observation in dataset.observations:
        observation_id = observation["observation_id"]
        path = feature_dir / f"{observation_id}.pt"
        if not path.is_file():
            raise FileNotFoundError(f"Missing feature file: {path}")
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload.get("observation_id") != observation_id:
            raise ValueError(f"Feature identity mismatch in {path}")
        payloads[observation_id] = payload
    return payloads


def grounding_embedding(payload: dict[str, Any]) -> torch.Tensor:
    """Summarize grounded patches, layout, and action probabilities."""
    patch_features = payload["patch_features"].to(dtype=torch.float32)
    logits = payload["element_logits"].to(dtype=torch.float32)
    patch_rows, patch_columns = payload["patch_grid"]
    if patch_features.shape[0] != patch_rows * patch_columns:
        raise ValueError("Grounding patch features do not match patch_grid")

    grid = patch_features.reshape(patch_rows, patch_columns, -1)
    row_midpoint = max(1, patch_rows // 2)
    column_midpoint = max(1, patch_columns // 2)
    quadrants = (
        grid[:row_midpoint, :column_midpoint],
        grid[:row_midpoint, column_midpoint:],
        grid[row_midpoint:, :column_midpoint],
        grid[row_midpoint:, column_midpoint:],
    )
    quadrant_means = [
        quadrant.reshape(-1, patch_features.shape[-1]).mean(dim=0)
        if quadrant.numel()
        else torch.zeros(patch_features.shape[-1])
        for quadrant in quadrants
    ]
    return torch.cat(
        [
            payload["global_embedding"].to(dtype=torch.float32),
            patch_features.mean(dim=0),
            patch_features.std(dim=0, unbiased=False),
            torch.cat(quadrant_means),
            torch.sigmoid(logits).mean(dim=0),
        ]
    )


def _prepare_element_payload(
    payload: dict[str, Any], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Validate and move one screen's normalized tile features and probabilities."""
    patch_features = payload.get("patch_features")
    element_logits = payload.get("element_logits")
    if not isinstance(patch_features, torch.Tensor) or patch_features.ndim != 2:
        raise ValueError("patch_features must have shape (tiles, embedding_dim)")
    if patch_features.shape[0] == 0 or patch_features.shape[1] == 0:
        raise ValueError("patch_features must not be empty")
    if (
        not isinstance(element_logits, torch.Tensor)
        or element_logits.ndim != 2
        or element_logits.shape
        != (patch_features.shape[0], len(ELEMENT_CLASS_NAMES))
    ):
        raise ValueError(
            "element_logits must have shape "
            f"(tiles, {len(ELEMENT_CLASS_NAMES)}) matching patch_features"
        )
    patch_grid = payload.get("patch_grid")
    if (
        not isinstance(patch_grid, (tuple, list))
        or len(patch_grid) != 2
        or not all(isinstance(size, int) and size > 0 for size in patch_grid)
        or patch_grid[0] * patch_grid[1] != patch_features.shape[0]
    ):
        raise ValueError(
            "patch_grid must contain positive rows and columns matching tiles"
        )
    if not torch.isfinite(patch_features).all():
        raise ValueError("patch_features must contain only finite values")
    if not torch.isfinite(element_logits).all():
        raise ValueError("element_logits must contain only finite values")

    normalized_features = F.normalize(
        patch_features.to(device=device, dtype=torch.float32),
        dim=1,
    )
    probabilities = torch.sigmoid(
        element_logits.to(device=device, dtype=torch.float32)
    )
    return normalized_features, probabilities


def _raw_element_matching_scores(
    first: tuple[torch.Tensor, torch.Tensor],
    second: tuple[torch.Tensor, torch.Tensor],
    *,
    tile_chunk_size: int,
) -> torch.Tensor:
    """Return one symmetric bidirectional best-match score per element class."""
    if tile_chunk_size < 1:
        raise ValueError("tile_chunk_size must be positive")
    first_features, first_probabilities = first
    second_features, second_probabilities = second
    if first_features.device != second_features.device:
        raise ValueError("Both screens must be on the same similarity device")
    if first_features.shape[1] != second_features.shape[1]:
        raise ValueError("Tile embedding dimensions must match")

    first_maxima = torch.zeros_like(first_probabilities)
    second_maxima = torch.zeros_like(second_probabilities)
    for first_start in range(0, first_features.shape[0], tile_chunk_size):
        first_end = first_start + tile_chunk_size
        first_feature_chunk = first_features[first_start:first_end]
        first_probability_chunk = first_probabilities[first_start:first_end]
        for second_start in range(0, second_features.shape[0], tile_chunk_size):
            second_end = second_start + tile_chunk_size
            second_probability_chunk = second_probabilities[
                second_start:second_end
            ]
            cosine_similarity = (
                first_feature_chunk
                @ second_features[second_start:second_end].T
            ).clamp_min(0)
            for class_index in range(len(ELEMENT_CLASS_NAMES)):
                weighted_similarity = (
                    cosine_similarity
                    * first_probability_chunk[:, class_index, None]
                    * second_probability_chunk[None, :, class_index]
                )
                first_maxima[
                    first_start:first_end, class_index
                ] = torch.maximum(
                    first_maxima[first_start:first_end, class_index],
                    weighted_similarity.max(dim=1).values,
                )
                second_maxima[
                    second_start:second_end, class_index
                ] = torch.maximum(
                    second_maxima[second_start:second_end, class_index],
                    weighted_similarity.max(dim=0).values,
                )

    epsilon = torch.finfo(first_probabilities.dtype).eps
    first_directed = first_maxima.sum(dim=0) / first_probabilities.sum(
        dim=0
    ).clamp_min(epsilon)
    second_directed = second_maxima.sum(dim=0) / second_probabilities.sum(
        dim=0
    ).clamp_min(epsilon)
    return (first_directed + second_directed) / 2


@torch.no_grad()
def element_matching_distance_matrix(
    payloads: Sequence[dict[str, Any]],
    *,
    class_weights: Sequence[float] = (1.0, 1.0),
    device: torch.device | str = "cpu",
    tile_chunk_size: int = 512,
) -> torch.Tensor:
    """Build exact pairwise screen distances from probability-weighted tile matches."""
    if len(class_weights) != len(ELEMENT_CLASS_NAMES):
        raise ValueError(
            f"class_weights must contain {len(ELEMENT_CLASS_NAMES)} values"
        )
    weights = torch.tensor(class_weights, dtype=torch.float32)
    if not torch.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("class_weights must be finite and non-negative")
    if weights.sum() <= 0:
        raise ValueError("At least one class weight must be positive")
    if tile_chunk_size < 1:
        raise ValueError("tile_chunk_size must be positive")

    similarity_device = torch.device(device)
    prepared = [
        _prepare_element_payload(payload, similarity_device) for payload in payloads
    ]
    observation_count = len(prepared)
    distances = torch.zeros(
        (observation_count, observation_count), dtype=torch.float32
    )
    if observation_count < 2:
        return distances

    self_scores = [
        _raw_element_matching_scores(
            screen,
            screen,
            tile_chunk_size=tile_chunk_size,
        )
        for screen in prepared
    ]
    device_weights = weights.to(similarity_device)
    epsilon = torch.finfo(torch.float32).eps
    for first_index, first in enumerate(prepared):
        for second_index in range(first_index):
            cross_score = _raw_element_matching_scores(
                first,
                prepared[second_index],
                tile_chunk_size=tile_chunk_size,
            )
            normalization = torch.sqrt(
                self_scores[first_index] * self_scores[second_index]
            ).clamp_min(epsilon)
            class_similarities = (cross_score / normalization).clamp(0, 1)
            similarity = (
                (class_similarities * device_weights).sum()
                / device_weights.sum()
            )
            distance = (1 - similarity).clamp(0, 1).cpu()
            distances[first_index, second_index] = distance
            distances[second_index, first_index] = distance
    return distances


def transition_embeddings(
    dataset: StateDataset,
    base_embeddings: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Append outgoing-neighbor and action statistics to visual embeddings."""
    event_types = sorted(
        {
            str(transition.get("event", {}).get("event_type", "unknown"))
            for transition in dataset.transitions
        }
    )
    event_index = {event_type: index for index, event_type in enumerate(event_types)}
    rows: list[torch.Tensor] = []
    for observation in dataset.observations:
        observation_id = observation["observation_id"]
        outgoing = dataset.outgoing_transitions(observation_id)
        destination_embeddings = [
            base_embeddings[transition["destination_observation_id"]]
            for transition in outgoing
        ]
        if destination_embeddings:
            neighbor_mean = torch.stack(destination_embeddings).mean(dim=0)
        else:
            neighbor_mean = torch.zeros_like(base_embeddings[observation_id])
        event_histogram = torch.zeros(len(event_types), dtype=torch.float32)
        for transition in outgoing:
            event_type = str(
                transition.get("event", {}).get("event_type", "unknown")
            )
            event_histogram[event_index[event_type]] += 1
        if outgoing:
            event_histogram /= len(outgoing)
        effective_ratio = (
            sum(bool(item.get("droidbot_effective")) for item in outgoing)
            / len(outgoing)
            if outgoing
            else 0.0
        )
        rows.append(
            torch.cat(
                [
                    base_embeddings[observation_id],
                    neighbor_mean,
                    event_histogram,
                    torch.tensor([len(outgoing), effective_ratio]),
                ]
            )
        )
    return torch.stack(rows)


def write_assignments(
    output_path: Path,
    dataset: StateDataset,
    baseline: str,
    labels: Sequence[int],
) -> None:
    if len(labels) != len(dataset.observations):
        raise ValueError("Every observation must receive exactly one cluster label")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(output_path.name + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as output:
        for observation, label in zip(dataset.observations, labels, strict=True):
            output.write(
                json.dumps(
                    {
                        "observation_id": observation["observation_id"],
                        "baseline": baseline,
                        "auto_cluster_id": f"{baseline}_{label:04d}",
                    }
                )
                + "\n"
            )
    os.replace(temporary_path, output_path)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument(
        "--baseline",
        required=True,
        choices=(
            "activity",
            "state_str",
            "structure_str",
            "perceptual",
            "dino",
            "grounding",
            "grounding_transition",
            "element_matching",
        ),
    )
    parser.add_argument("--feature-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("state_clusters"))
    parser.add_argument("--distance-threshold", type=float, default=0.15)
    parser.add_argument("--max-hamming-distance", type=int, default=8)
    parser.add_argument("--clickable-weight", type=float, default=1.0)
    parser.add_argument("--scrollable-weight", type=float, default=1.0)
    parser.add_argument(
        "--similarity-device",
        default="auto",
        help="Torch device for tile matching (default: CUDA when available).",
    )
    parser.add_argument("--tile-chunk-size", type=int, default=512)
    return parser.parse_args(argv)


def resolve_similarity_device(name: str) -> torch.device:
    """Resolve an explicit or automatically selected tile-similarity device."""
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA similarity device requested, but CUDA is unavailable")
    return device


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    dataset = StateDataset.load(args.run_dir)
    baseline = args.baseline
    field_by_baseline = {
        "activity": "activity",
        "state_str": "droidbot_state_str",
        "structure_str": "droidbot_structure_str",
    }
    if baseline in field_by_baseline:
        labels = categorical_clusters(
            dataset.observations,
            field_by_baseline[baseline],
        )
    elif baseline == "perceptual":
        if args.feature_dir is None:
            raise ValueError("--feature-dir must contain perceptual_hashes.jsonl")
        hashes_by_id = {
            record["observation_id"]: record["feature"]
            for record in read_jsonl(args.feature_dir / "perceptual_hashes.jsonl")
        }
        labels = perceptual_hash_clusters(
            [
                hashes_by_id[observation["observation_id"]]
                for observation in dataset.observations
            ],
            max_hamming_distance=args.max_hamming_distance,
        )
    else:
        if args.feature_dir is None:
            raise ValueError("--feature-dir is required for embedding baselines")
        payloads = load_feature_payloads(dataset, args.feature_dir)
        if baseline == "element_matching":
            distances = element_matching_distance_matrix(
                [
                    payloads[observation["observation_id"]]
                    for observation in dataset.observations
                ],
                class_weights=(
                    args.clickable_weight,
                    args.scrollable_weight,
                ),
                device=resolve_similarity_device(args.similarity_device),
                tile_chunk_size=args.tile_chunk_size,
            )
            labels = distance_matrix_clusters(
                distances,
                distance_threshold=args.distance_threshold,
            )
        elif baseline == "dino":
            embeddings = torch.stack(
                [
                    payloads[observation["observation_id"]]["global_embedding"]
                    for observation in dataset.observations
                ]
            )
        else:
            base_by_id = {
                observation_id: grounding_embedding(payload)
                for observation_id, payload in payloads.items()
            }
            if baseline == "grounding_transition":
                embeddings = transition_embeddings(dataset, base_by_id)
            else:
                embeddings = torch.stack(
                    [
                        base_by_id[observation["observation_id"]]
                        for observation in dataset.observations
                    ]
                )
        if baseline != "element_matching":
            labels = embedding_clusters(
                embeddings,
                distance_threshold=args.distance_threshold,
            )

    output_path = args.output_dir / dataset.run_id / f"{baseline}.jsonl"
    write_assignments(output_path, dataset, baseline, labels)
    cluster_count = len(set(labels))
    print(
        f"Wrote {len(labels)} assignments in {cluster_count} clusters to "
        f"{output_path}"
    )


if __name__ == "__main__":
    main()
