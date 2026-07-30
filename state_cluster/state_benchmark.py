"""Compare automatic and manually corrected cluster annotation files."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

from state_dataset import StateDataset, load_annotations, validate_annotations


def pairwise_metrics(
    ground_truth: Sequence[Any], predicted: Sequence[Any]
) -> dict[str, float | int]:
    if len(ground_truth) != len(predicted):
        raise ValueError("Ground-truth and predicted labels must have equal length")
    truth_counts = Counter(ground_truth)
    prediction_counts = Counter(predicted)
    joint_counts = Counter(zip(ground_truth, predicted, strict=True))

    def pair_count(count: int) -> int:
        return count * (count - 1) // 2

    true_positive = sum(pair_count(count) for count in joint_counts.values())
    predicted_positive = sum(
        pair_count(count) for count in prediction_counts.values()
    )
    actual_positive = sum(pair_count(count) for count in truth_counts.values())
    total_pairs = pair_count(len(ground_truth))
    false_positive = predicted_positive - true_positive
    false_negative = actual_positive - true_positive
    true_negative = (
        total_pairs - true_positive - false_positive - false_negative
    )
    precision = (
        true_positive / (true_positive + false_positive)
        if true_positive + false_positive
        else 0.0
    )
    recall = (
        true_positive / (true_positive + false_negative)
        if true_positive + false_negative
        else 0.0
    )
    pairwise_f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return {
        "pairwise_precision": precision,
        "pairwise_recall": recall,
        "pairwise_f1": pairwise_f1,
        "false_merge_rate": (
            false_positive / (true_positive + false_positive)
            if true_positive + false_positive
            else 0.0
        ),
        "false_split_rate": (
            false_negative / (true_positive + false_negative)
            if true_positive + false_negative
            else 0.0
        ),
        "true_positive_pairs": true_positive,
        "false_positive_pairs": false_positive,
        "false_negative_pairs": false_negative,
        "true_negative_pairs": true_negative,
    }


def manual_corrections_required(
    ground_truth: Sequence[str], predicted: Sequence[str]
) -> int:
    """Count observations differing from each predicted cluster's majority."""
    labels_by_cluster: dict[str, list[str]] = defaultdict(list)
    for truth, prediction in zip(ground_truth, predicted, strict=True):
        labels_by_cluster[prediction].append(truth)
    return sum(
        len(labels) - Counter(labels).most_common(1)[0][1]
        for labels in labels_by_cluster.values()
    )


def evaluate_assignments(
    dataset: StateDataset,
    assignments: Sequence[dict[str, str]],
    reference: Sequence[dict[str, str]],
) -> dict[str, Any]:
    """Evaluate the overlap of two validated assignment annotations."""
    assignments = validate_annotations(dataset, assignments)
    reference = validate_annotations(dataset, reference)
    predicted_by_id = {
        record["observation_id"]: record["cluster_id"]
        for record in assignments
    }
    reference_by_id = {
        record["observation_id"]: record["cluster_id"] for record in reference
    }
    observation_ids = [
        observation["observation_id"]
        for observation in dataset.observations
        if observation["observation_id"] in predicted_by_id
        and observation["observation_id"] in reference_by_id
    ]
    if len(observation_ids) < 2:
        raise ValueError(
            "At least two observations shared by both annotations are required"
        )
    predicted = [predicted_by_id[observation_id] for observation_id in observation_ids]
    ground_truth = [
        reference_by_id[observation_id] for observation_id in observation_ids
    ]

    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

    result: dict[str, Any] = {
        "run_id": dataset.run_id,
        "predicted_source": assignments[0]["source"] if assignments else None,
        "reference_source": reference[0]["source"] if reference else None,
        "evaluated_observations": len(observation_ids),
        "reference_clusters": len(set(ground_truth)),
        "predicted_clusters": len(set(predicted)),
        "adjusted_rand_index": adjusted_rand_score(ground_truth, predicted),
        "normalized_mutual_information": normalized_mutual_info_score(
            ground_truth, predicted
        ),
        "manual_corrections_required": manual_corrections_required(
            ground_truth, predicted
        ),
    }
    result.update(pairwise_metrics(ground_truth, predicted))
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("assignments", type=Path)
    parser.add_argument("reference", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    dataset = StateDataset.load(args.run_dir)
    result = evaluate_assignments(
        dataset,
        load_annotations(dataset, args.assignments),
        load_annotations(dataset, args.reference),
    )
    rendered = json.dumps(result, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
        print(f"Wrote {args.output}")
    else:
        print(rendered)


if __name__ == "__main__":
    main()
