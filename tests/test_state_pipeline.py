import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image
from torchvision.io import ImageReadMode, read_image

from state_annotation_app import (
    _default_merged_reviews_path,
    build_cluster_groups,
    flagged_observations,
    merge_cluster_assignments,
    ordered_cluster_ids,
    parse_args,
    representative_observation_ids,
    review_is_current,
    save_cluster_review,
    write_cluster_assignments,
)
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
    def test_default_merged_reviews_path_follows_normal_reviews(self):
        args = type("Args", (), {"merged_reviews": None})()
        self.assertEqual(
            Path("annotations/run_reviews_merged.jsonl"),
            _default_merged_reviews_path(
                args,
                Path("annotations/run_reviews.jsonl"),
            ),
        )

    def test_explicit_merged_reviews_path_is_preserved(self):
        output_path = Path("custom/focused_reviews.jsonl")
        args = type("Args", (), {"merged_reviews": output_path})()
        self.assertEqual(
            output_path,
            _default_merged_reviews_path(
                args,
                Path("annotations/run_reviews.jsonl"),
            ),
        )

    def test_parse_args_accepts_merged_reviews_output(self):
        args = parse_args(
            [
                "--run-dir",
                "run_001",
                "--clusters",
                "clusters.jsonl",
                "--merged-reviews",
                "focused_reviews.jsonl",
            ]
        )
        self.assertEqual(Path("focused_reviews.jsonl"), args.merged_reviews)

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

    def test_cluster_review_writes_replace_records_by_cluster(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            review_path = Path(temp_dir) / "reviews.jsonl"
            records = {}
            save_cluster_review(
                review_path,
                records,
                "cluster_0001",
                ["obs_000001", "obs_000002"],
                ["obs_000002"],
            )
            save_cluster_review(
                review_path,
                records,
                "cluster_0001",
                ["obs_000001", "obs_000002"],
                [],
            )
            reviews = read_jsonl(review_path)
            self.assertEqual(1, len(reviews))
            self.assertEqual([], reviews[0]["incorrect_observation_ids"])
            self.assertTrue(
                review_is_current(
                    reviews[0],
                    ["obs_000001", "obs_000002"],
                )
            )
            self.assertFalse(
                review_is_current(
                    reviews[0],
                    ["obs_000001", "obs_000002", "obs_000003"],
                )
            )

    def test_merged_reviews_do_not_modify_original_reviews(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            original_path = Path(temp_dir) / "reviews.jsonl"
            merged_path = Path(temp_dir) / "reviews_merged.jsonl"
            original_reviews = {}
            merged_reviews = {}
            save_cluster_review(
                original_path,
                original_reviews,
                "cluster_0001",
                ["obs_000001"],
                [],
            )
            save_cluster_review(
                merged_path,
                merged_reviews,
                "cluster_0001",
                ["obs_000001", "obs_000002"],
                ["obs_000002"],
            )

            self.assertEqual(
                ["obs_000001"],
                read_jsonl(original_path)[0]["observation_ids"],
            )
            self.assertEqual(
                ["obs_000001", "obs_000002"],
                read_jsonl(merged_path)[0]["observation_ids"],
            )

    def test_cluster_review_orders_multi_state_groups_before_singletons(self):
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
                "observation_id": "obs_000001",
                "auto_cluster_id": "singleton_a",
            },
            {
                "observation_id": "obs_000002",
                "auto_cluster_id": "group_b",
            },
            {
                "observation_id": "obs_000003",
                "auto_cluster_id": "group_b",
            },
            {
                "observation_id": "obs_000004",
                "auto_cluster_id": "singleton_c",
            },
        ]
        groups = build_cluster_groups(dataset, assignments)
        cluster_ids = ordered_cluster_ids(groups)
        self.assertEqual(
            ["group_b", "singleton_a", "singleton_c"],
            cluster_ids,
        )

        reviews = {
            "group_b": {
                "cluster_id": "group_b",
                "observation_ids": ["obs_000002", "obs_000003"],
                "incorrect_observation_ids": ["obs_000003"],
                "status": "confirmed",
            },
            "singleton_a": {
                "cluster_id": "singleton_a",
                "observation_ids": ["obs_000001"],
                "incorrect_observation_ids": ["obs_000001"],
                "status": "confirmed",
            },
        }
        self.assertEqual(
            {
                "group_b": ("obs_000003",),
                "singleton_a": ("obs_000001",),
            },
            flagged_observations(cluster_ids, groups, reviews),
        )

    def test_cluster_review_rejects_missing_assignments(self):
        observation = {"observation_id": "obs_000001"}
        dataset = StateDataset(
            run_dir=Path("."),
            manifest={"run_id": "run_001"},
            observations=(observation,),
            transitions=(),
            observations_by_id={"obs_000001": observation},
        )
        with self.assertRaisesRegex(ValueError, "Missing cluster assignments"):
            build_cluster_groups(dataset, [])

    def test_cluster_review_rejects_flags_outside_cluster(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaisesRegex(
                ValueError,
                "must belong to the cluster",
            ):
                save_cluster_review(
                    Path(temp_dir) / "reviews.jsonl",
                    {},
                    "cluster_0001",
                    ["obs_000001"],
                    ["obs_000002"],
                )

    def test_merge_cluster_assignments_uses_smallest_selected_id(self):
        assignments = [
            {
                "observation_id": "obs_000001",
                "baseline": "structure_str",
                "auto_cluster_id": "cluster_b",
            },
            {
                "observation_id": "obs_000002",
                "baseline": "structure_str",
                "auto_cluster_id": "cluster_a",
            },
            {
                "observation_id": "obs_000003",
                "baseline": "structure_str",
                "auto_cluster_id": "cluster_c",
            },
        ]
        merged, cluster_id = merge_cluster_assignments(
            assignments,
            ["cluster_b", "cluster_a"],
        )
        self.assertEqual("cluster_a", cluster_id)
        self.assertEqual(
            ["cluster_a", "cluster_a", "cluster_c"],
            [record["auto_cluster_id"] for record in merged],
        )
        self.assertEqual("cluster_b", assignments[0]["auto_cluster_id"])

    def test_merge_cluster_assignments_supports_transitive_merges(self):
        assignments = [
            {
                "observation_id": f"obs_{index:06d}",
                "baseline": "structure_str",
                "auto_cluster_id": cluster_id,
            }
            for index, cluster_id in enumerate(
                ["cluster_a", "cluster_b", "cluster_c"],
                start=1,
            )
        ]
        first_merge, _ = merge_cluster_assignments(
            assignments,
            ["cluster_a", "cluster_b"],
        )
        second_merge, _ = merge_cluster_assignments(
            first_merge,
            ["cluster_a", "cluster_c"],
        )
        self.assertEqual(
            ["cluster_a", "cluster_a", "cluster_a"],
            [record["auto_cluster_id"] for record in second_merge],
        )

    def test_merged_assignments_use_pipeline_jsonl_schema(self):
        assignments = [
            {
                "observation_id": "obs_000001",
                "baseline": "structure_str",
                "auto_cluster_id": "cluster_a",
                "ignored_extra_field": True,
            }
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "merged.jsonl"
            write_cluster_assignments(output_path, assignments)
            records = read_jsonl(output_path)
        self.assertEqual(
            [
                {
                    "observation_id": "obs_000001",
                    "baseline": "structure_str",
                    "auto_cluster_id": "cluster_a",
                }
            ],
            records,
        )

    def test_representative_observations_cover_cluster_range(self):
        observation_ids = [f"obs_{index:06d}" for index in range(1, 11)]
        self.assertEqual(
            ("obs_000001", "obs_000005", "obs_000010"),
            representative_observation_ids(observation_ids),
        )
        self.assertEqual(
            ("obs_000001",),
            representative_observation_ids(observation_ids, maximum=1),
        )


if __name__ == "__main__":
    unittest.main()
