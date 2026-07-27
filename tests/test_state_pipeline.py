import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image
from torchvision.io import ImageReadMode, read_image

from state_annotation_app import save_annotations, save_pair_decision
from state_benchmark import (
    evaluate_assignments,
    manual_corrections_required,
    pairwise_metrics,
)
from state_clustering import (
    categorical_clusters,
    embedding_clusters,
    perceptual_hash_clusters,
)
from state_dataset import StateDataset, read_jsonl
from state_features import difference_hash, hamming_distance


class StatePipelineTest(unittest.TestCase):
    def test_difference_hash_is_deterministic_and_64_bits(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "screen.png"
            Image.new("RGB", (432, 768), color=(12, 34, 56)).save(image_path)
            image = read_image(str(image_path), mode=ImageReadMode.RGB)
            first_hash = difference_hash(image)
            second_hash = difference_hash(image.clone())
        self.assertEqual(16, len(first_hash))
        self.assertEqual(first_hash, second_hash)
        self.assertEqual(0, hamming_distance(first_hash, second_hash))

    def test_baseline_cluster_helpers(self):
        observations = [
            {"activity": "A"},
            {"activity": "B"},
            {"activity": "A"},
        ]
        self.assertEqual([0, 1, 0], categorical_clusters(observations, "activity"))
        self.assertEqual(
            [0, 0, 1],
            perceptual_hash_clusters(
                ["0000000000000000", "0000000000000001", "ffffffffffffffff"],
                max_hamming_distance=1,
            ),
        )
        labels = embedding_clusters(
            torch.tensor(
                [
                    [1.0, 0.0],
                    [0.99, 0.01],
                    [0.0, 1.0],
                ]
            ),
            distance_threshold=0.1,
        )
        self.assertEqual(labels[0], labels[1])
        self.assertNotEqual(labels[0], labels[2])

    def test_pairwise_metrics_and_manual_corrections(self):
        ground_truth = ["A", "A", "B", "B"]
        predicted = ["X", "X", "X", "Y"]
        metrics = pairwise_metrics(ground_truth, predicted)
        self.assertAlmostEqual(1 / 3, metrics["pairwise_precision"])
        self.assertAlmostEqual(1 / 2, metrics["pairwise_recall"])
        self.assertEqual(
            1,
            manual_corrections_required(ground_truth, predicted),
        )

    def test_evaluate_assignments_uses_manual_labels(self):
        observations = tuple(
            {"observation_id": f"obs_{index:06d}"} for index in range(1, 5)
        )
        dataset = StateDataset(
            run_dir=Path("."),
            manifest={"run_id": "run_001"},
            observations=observations,
            transitions=(),
            observations_by_id={
                observation["observation_id"]: observation
                for observation in observations
            },
        )
        assignments = [
            {
                "observation_id": observation["observation_id"],
                "baseline": "test",
                "auto_cluster_id": "A" if index < 2 else "B",
            }
            for index, observation in enumerate(observations)
        ]
        annotations = [
            {
                "observation_id": observation["observation_id"],
                "manual_functional_state_label": "A" if index < 2 else "B",
                "status": "valid",
            }
            for index, observation in enumerate(observations)
        ]
        result = evaluate_assignments(dataset, assignments, annotations)
        self.assertEqual(1.0, result["adjusted_rand_index"])
        self.assertEqual(1.0, result["pairwise_f1"])

    def test_annotation_writes_replace_records_by_identity(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            annotation_path = Path(temp_dir) / "annotations.jsonl"
            records = {}
            save_annotations(
                annotation_path,
                records,
                ["obs_000001", "obs_000002"],
                functional_label="Article",
                substate_label="",
                status="valid",
            )
            save_annotations(
                annotation_path,
                records,
                ["obs_000002"],
                functional_label="Dialog",
                substate_label="",
                status="valid",
            )
            annotations = read_jsonl(annotation_path)
            self.assertEqual(2, len(annotations))
            self.assertEqual(
                "Dialog",
                {
                    item["observation_id"]: item for item in annotations
                }["obs_000002"]["manual_functional_state_label"],
            )

            pairs_path = Path(temp_dir) / "pairs.jsonl"
            pairs = {}
            save_pair_decision(
                pairs_path,
                pairs,
                "obs_000002",
                "obs_000001",
                "same",
            )
            self.assertEqual(
                "obs_000001__obs_000002",
                read_jsonl(pairs_path)[0]["pair_id"],
            )


if __name__ == "__main__":
    unittest.main()
