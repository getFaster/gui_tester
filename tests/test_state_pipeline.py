import json
import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image
from streamlit.testing.v1 import AppTest
from torchvision.io import ImageReadMode, read_image

from state_annotation_app import (
    _default_merged_reviews_path,
    assign_outliers_to_same_cluster,
    assign_outliers_to_singleton_clusters,
    build_cluster_groups,
    current_outlier_observation_ids,
    flagged_observations,
    merge_cluster_assignments,
    next_unreviewed_cluster_id,
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


def run_annotation_app_for_test(
    run_dir: str,
    clusters_path: str,
    reviews_path: str,
    merged_clusters_path: str,
    merged_reviews_path: str,
) -> None:
    """Run the annotation app with isolated test output paths."""
    from state_annotation_app import main

    main(
        [
            "--run-dir",
            run_dir,
            "--clusters",
            clusters_path,
            "--reviews",
            reviews_path,
            "--merged-clusters",
            merged_clusters_path,
            "--merged-reviews",
            merged_reviews_path,
        ]
    )


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

    def test_annotation_ui_saves_exact_visible_outlier_selection(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            run_dir = root / "run_001"
            screenshots_dir = run_dir / "screenshots"
            states_dir = run_dir / "states"
            screenshots_dir.mkdir(parents=True)
            states_dir.mkdir()

            observation_ids = ["obs_000001", "obs_000002"]
            observations = []
            for index, observation_id in enumerate(observation_ids):
                screenshot_relative_path = (
                    Path("screenshots") / f"{observation_id}.png"
                )
                state_relative_path = Path("states") / f"{observation_id}.json"
                Image.new(
                    "RGB",
                    (432, 768),
                    color=(index * 32, 0, 0),
                ).save(run_dir / screenshot_relative_path)
                (run_dir / state_relative_path).write_text(
                    "{}",
                    encoding="utf-8",
                )
                observations.append(
                    {
                        "observation_id": observation_id,
                        "screenshot_path": screenshot_relative_path.as_posix(),
                        "view_tree_path": state_relative_path.as_posix(),
                        "activity": "org.example/.MainActivity",
                    }
                )

            (run_dir / "run.json").write_text(
                json.dumps({"run_id": "run_001"}),
                encoding="utf-8",
            )
            (run_dir / "observations.jsonl").write_text(
                "".join(
                    json.dumps(observation) + "\n"
                    for observation in observations
                ),
                encoding="utf-8",
            )
            (run_dir / "transitions.jsonl").write_text(
                json.dumps(
                    {
                        "transition_id": "trans_000001",
                        "source_observation_id": observation_ids[0],
                        "destination_observation_id": observation_ids[1],
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            clusters_path = root / "clusters.jsonl"
            clusters_path.write_text(
                "".join(
                    json.dumps(
                        {
                            "observation_id": observation_id,
                            "baseline": "test",
                            "auto_cluster_id": "cluster_0001",
                        }
                    )
                    + "\n"
                    for observation_id in observation_ids
                ),
                encoding="utf-8",
            )
            reviews_path = root / "reviews.jsonl"
            merged_clusters_path = root / "clusters_merged.jsonl"
            merged_reviews_path = root / "reviews_merged.jsonl"

            app = AppTest.from_function(
                run_annotation_app_for_test,
                args=(
                    str(run_dir),
                    str(clusters_path),
                    str(reviews_path),
                    str(merged_clusters_path),
                    str(merged_reviews_path),
                ),
            ).run()
            self.assertEqual([], app.exception)

            outlier_checkboxes = [
                checkbox
                for checkbox in app.checkbox
                if checkbox.label == "Does not belong"
            ]
            self.assertEqual(2, len(outlier_checkboxes))
            self.assertEqual(
                [False, False],
                [checkbox.value for checkbox in outlier_checkboxes],
            )

            outlier_checkboxes[1].check().run()
            visible_outlier_values = [
                checkbox.value
                for checkbox in app.checkbox
                if checkbox.label == "Does not belong"
            ]
            self.assertEqual([False, True], visible_outlier_values)

            confirm_button = next(
                button
                for button in app.button
                if button.label == "Confirm and continue"
            )
            confirm_button.click().run()
            self.assertEqual([], app.exception)

            self.assertEqual(
                [
                    {
                        "cluster_id": "cluster_0001",
                        "observation_ids": observation_ids,
                        "incorrect_observation_ids": [observation_ids[1]],
                        "status": "confirmed",
                    }
                ],
                read_jsonl(reviews_path),
            )
            merged_groups = build_cluster_groups(
                StateDataset.load(run_dir),
                read_jsonl(merged_clusters_path),
            )
            self.assertEqual(2, len(merged_groups))
            self.assertTrue(
                all(
                    len(group_observation_ids) == 1
                    for group_observation_ids in merged_groups.values()
                )
            )
            visible_outlier_checkboxes = [
                checkbox
                for checkbox in app.checkbox
                if checkbox.label == "Does not belong"
                and "original" in str(checkbox.key)
            ]
            self.assertEqual(1, len(visible_outlier_checkboxes))
            self.assertIn(
                observation_ids[0],
                str(visible_outlier_checkboxes[0].key),
            )
            self.assertNotIn(
                observation_ids[1],
                str(visible_outlier_checkboxes[0].key),
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

    def test_review_advances_to_next_unreviewed_cluster(self):
        cluster_ids = ["cluster_a", "cluster_b", "cluster_c", "cluster_d"]
        self.assertEqual(
            "cluster_d",
            next_unreviewed_cluster_id(
                cluster_ids,
                ["cluster_a", "cluster_d"],
                "cluster_b",
            ),
        )
        self.assertEqual(
            "cluster_a",
            next_unreviewed_cluster_id(
                cluster_ids,
                ["cluster_a"],
                "cluster_d",
            ),
        )
        self.assertIsNone(
            next_unreviewed_cluster_id(
                cluster_ids,
                [],
                "cluster_b",
            )
        )

    def test_merged_membership_requires_a_new_review(self):
        review = {
            "cluster_id": "cluster_a",
            "observation_ids": ["obs_000001"],
            "incorrect_observation_ids": [],
            "status": "confirmed",
        }
        self.assertTrue(review_is_current(review, ["obs_000001"]))
        self.assertFalse(
            review_is_current(
                review,
                ["obs_000001", "obs_000002"],
            )
        )

    def test_preview_outliers_include_both_current_review_sources(self):
        observation_ids = ["obs_000001", "obs_000002", "obs_000003"]
        original_reviews = {
            "cluster_a": {
                "cluster_id": "cluster_a",
                "observation_ids": observation_ids,
                "incorrect_observation_ids": ["obs_000001"],
                "status": "confirmed",
            }
        }
        merged_reviews = {
            "cluster_a": {
                "cluster_id": "cluster_a",
                "observation_ids": observation_ids,
                "incorrect_observation_ids": ["obs_000003"],
                "status": "confirmed",
            }
        }
        self.assertEqual(
            ("obs_000001", "obs_000003"),
            current_outlier_observation_ids(
                "cluster_a",
                observation_ids,
                (original_reviews, merged_reviews),
            ),
        )

    def test_preview_ignores_outliers_from_stale_membership(self):
        reviews = {
            "cluster_a": {
                "cluster_id": "cluster_a",
                "observation_ids": ["obs_000001"],
                "incorrect_observation_ids": ["obs_000001"],
                "status": "confirmed",
            }
        }
        self.assertEqual(
            (),
            current_outlier_observation_ids(
                "cluster_a",
                ["obs_000001", "obs_000002"],
                (reviews,),
            ),
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

    def test_annotated_outliers_become_distinct_singleton_clusters(self):
        assignments = [
            {
                "observation_id": observation_id,
                "baseline": "embedding",
                "auto_cluster_id": "cluster_a",
            }
            for observation_id in ("state_1", "state_2", "state_3")
        ]

        updated = assign_outliers_to_singleton_clusters(
            assignments,
            ["state_1", "state_2"],
        )
        groups: dict[str, list[str]] = {}
        for record in updated:
            groups.setdefault(record["auto_cluster_id"], []).append(
                record["observation_id"]
            )

        self.assertEqual(3, len(groups))
        self.assertTrue(
            all(len(observation_ids) == 1 for observation_ids in groups.values())
        )
        self.assertEqual("cluster_a", updated[2]["auto_cluster_id"])
        self.assertTrue(
            updated[0]["auto_cluster_id"].startswith("cluster_a__outlier_")
        )
        self.assertNotEqual(
            updated[0]["auto_cluster_id"],
            updated[1]["auto_cluster_id"],
        )
        self.assertTrue(
            all(
                record["auto_cluster_id"] == "cluster_a"
                for record in assignments
            )
        )

    def test_selected_outliers_can_become_one_new_cluster(self):
        assignments = [
            {
                "observation_id": observation_id,
                "baseline": "embedding",
                "auto_cluster_id": "cluster_a",
            }
            for observation_id in ("state_1", "state_2", "state_3")
        ]

        updated = assign_outliers_to_same_cluster(
            assignments,
            ["state_1", "state_2"],
        )

        self.assertEqual(
            updated[0]["auto_cluster_id"],
            updated[1]["auto_cluster_id"],
        )
        self.assertTrue(
            updated[0]["auto_cluster_id"].startswith("outlier_group_")
        )
        self.assertEqual("cluster_a", updated[2]["auto_cluster_id"])
        self.assertTrue(
            all(
                record["auto_cluster_id"] == "cluster_a"
                for record in assignments
            )
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
