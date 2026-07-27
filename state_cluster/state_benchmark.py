"""Evaluate automatic state clusters against manual functional-state labels."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

from state_dataset import StateDataset, read_jsonl


def pairwise_metrics(
    ground_truth: Sequence[str], predicted: Sequence[str]
) -> dict[str, float | int]:
    if len(ground_truth) != len(predicted):
        raise ValueError("Ground-truth and predicted labels must have equal length")
    true_positive = false_positive = false_negative = true_negative = 0
    for first_index in range(len(ground_truth)):
        for second_index in range(first_index):
            same_truth = (
                ground_truth[first_index] == ground_truth[second_index]
            )
            same_prediction = predicted[first_index] == predicted[second_index]
            if same_truth and same_prediction:
                true_positive += 1
            elif same_prediction:
                false_positive += 1
            elif same_truth:
                false_negative += 1
            else:
                true_negative += 1

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
    """Count observations that differ from each predicted cluster's majority."""
    labels_by_cluster: dict[str, list[str]] = defaultdict(list)
    for truth, prediction in zip(ground_truth, predicted, strict=True):
        labels_by_cluster[prediction].append(truth)
    return sum(
        len(labels) - Counter(labels).most_common(1)[0][1]
        for labels in labels_by_cluster.values()
    )


def evaluate_assignments(
    dataset: StateDataset,
    assignments: Sequence[dict[str, Any]],
    annotations: Sequence[dict[str, Any]],
    *,
    include_substate: bool = False,
) -> dict[str, Any]:
    assignment_by_id = {
        record["observation_id"]: record["auto_cluster_id"]
        for record in assignments
    }
    annotation_by_id = {
        record["observation_id"]: record for record in annotations
    }
    observation_ids: list[str] = []
    ground_truth: list[str] = []
    predicted: list[str] = []
    for observation in dataset.observations:
        observation_id = observation["observation_id"]
        annotation = annotation_by_id.get(observation_id)
        if annotation is None or annotation.get("status") == "invalid":
            continue
        functional_label = annotation.get("manual_functional_state_label")
        if not functional_label or observation_id not in assignment_by_id:
            continue
        label = str(functional_label)
        if include_substate and annotation.get("manual_substate_label"):
            label += "::" + str(annotation["manual_substate_label"])
        observation_ids.append(observation_id)
        ground_truth.append(label)
        predicted.append(assignment_by_id[observation_id])

    if len(observation_ids) < 2:
        raise ValueError("At least two labeled observations are required")

    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

    result: dict[str, Any] = {
        "run_id": dataset.run_id,
        "baseline": assignments[0].get("baseline") if assignments else None,
        "include_substate": include_substate,
        "evaluated_observations": len(observation_ids),
        "ground_truth_clusters": len(set(ground_truth)),
        "predicted_clusters": len(set(predicted)),
        "adjusted_rand_index": adjusted_rand_score(ground_truth, predicted),
        "normalized_mutual_information": normalized_mutual_info_score(
            ground_truth,
            predicted,
        ),
        "manual_corrections_required": manual_corrections_required(
            ground_truth,
            predicted,
        ),
    }
    result.update(pairwise_metrics(ground_truth, predicted))
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("assignments", type=Path)
    parser.add_argument("annotations", type=Path)
    parser.add_argument("--include-substate", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    dataset = StateDataset.load(args.run_dir)
    result = evaluate_assignments(
        dataset,
        read_jsonl(args.assignments),
        read_jsonl(args.annotations),
        include_substate=args.include_substate,
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
