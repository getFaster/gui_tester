import csv
import json
import math
import tempfile
import unittest
from pathlib import Path

import torch

from state_cluster.state_threshold_sweep import (
    best_row,
    deduplicated_observation_ids,
    evaluate_clusterings,
    exhaustive_average_linkage_clusterings,
    main,
    metric_values,
    row_at_threshold,
    write_metric_plot,
    write_sweep_csv,
)


class StateThresholdSweepTest(unittest.TestCase):
    def setUp(self) -> None:
        self.distances = torch.tensor(
            [
                [0.0, 0.1, 0.8],
                [0.1, 0.0, 0.8],
                [0.8, 0.8, 0.0],
            ]
        )

    def test_enumerates_every_effective_average_linkage_threshold(self) -> None:
        clusterings = exhaustive_average_linkage_clusterings(self.distances)

        self.assertEqual(
            [3, 2, 1], [clustering["predicted_clusters"] for clustering in clusterings]
        )
        self.assertEqual(0.0, clusterings[0]["distance_threshold"])
        self.assertEqual(
            math.nextafter(clusterings[1]["merge_distance"], math.inf),
            clusterings[1]["distance_threshold"],
        )
        self.assertEqual(clusterings[1]["labels"][0], clusterings[1]["labels"][1])
        self.assertNotEqual(clusterings[1]["labels"][0], clusterings[1]["labels"][2])

    def test_equal_merge_distances_do_not_create_unreachable_cut(self) -> None:
        distances = torch.tensor(
            [
                [0.0, 0.5, 0.5],
                [0.5, 0.0, 0.5],
                [0.5, 0.5, 0.0],
            ]
        )

        clusterings = exhaustive_average_linkage_clusterings(distances)

        self.assertEqual(
            [3, 1],
            [clustering["predicted_clusters"] for clustering in clusterings],
        )

    def test_scores_full_and_deduplicated_views_and_selects_best(self) -> None:
        observation_ids = ["obs_1", "obs_2", "obs_3"]
        reference = [
            {"observation_id": "obs_1", "cluster_id": "A", "source": "manual"},
            {"observation_id": "obs_2", "cluster_id": "A", "source": "manual"},
            {"observation_id": "obs_3", "cluster_id": "B", "source": "manual"},
        ]
        clusterings = exhaustive_average_linkage_clusterings(self.distances)

        rows = evaluate_clusterings(
            clusterings,
            observation_ids,
            reference,
            observation_ids,
        )
        best = best_row(rows, view_name="full")

        self.assertEqual(2, best["predicted_clusters"])
        self.assertEqual(1.0, best["full_pairwise_f1"])
        self.assertEqual(1.0, best["full_adjusted_rand_index"])
        self.assertEqual(1.0, best["full_normalized_mutual_information"])
        self.assertEqual(rows[0], row_at_threshold(rows, 0.1))
        self.assertEqual(rows[1], row_at_threshold(rows, 0.2))

    def test_deduplicated_view_uses_first_eligible_group_member(self) -> None:
        groups = [
            {
                "pixel_sha256": "a" * 64,
                "observation_ids": ["obs_1", "obs_2"],
            },
            {
                "pixel_sha256": "b" * 64,
                "observation_ids": ["obs_3"],
            },
        ]

        selected = deduplicated_observation_ids(groups, {"obs_2", "obs_3"})

        self.assertEqual(("obs_2", "obs_3"), selected)

    def test_pairwise_f1_handles_integer_predictions(self) -> None:
        metrics = metric_values(["A", "A", "B"], [0, 0, 1])

        self.assertEqual(1.0, metrics["pairwise_f1"])

    def test_writes_machine_readable_sweep_and_svg(self) -> None:
        rows = evaluate_clusterings(
            exhaustive_average_linkage_clusterings(self.distances),
            ["obs_1", "obs_2", "obs_3"],
            [
                {
                    "observation_id": "obs_1",
                    "cluster_id": "A",
                    "source": "manual",
                },
                {
                    "observation_id": "obs_2",
                    "cluster_id": "A",
                    "source": "manual",
                },
                {
                    "observation_id": "obs_3",
                    "cluster_id": "B",
                    "source": "manual",
                },
            ],
            ["obs_1", "obs_2", "obs_3"],
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            csv_path = root / "threshold_sweep.csv"
            svg_path = root / "threshold_metrics.svg"

            write_sweep_csv(csv_path, rows)
            write_metric_plot(svg_path, rows)

            with csv_path.open(encoding="utf-8", newline="") as source:
                written_rows = list(csv.DictReader(source))
            self.assertEqual(len(rows), len(written_rows))
            self.assertIn("full_pairwise_f1", written_rows[0])
            self.assertIn("<svg", svg_path.read_text(encoding="utf-8"))

    def test_cli_writes_complete_experiment_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            run_dir = root / "run"
            feature_dir = root / "features"
            output_root = root / "output"
            (run_dir / "screenshots").mkdir(parents=True)
            (run_dir / "states").mkdir()
            feature_dir.mkdir()
            observation_ids = (
                "obs_000001",
                "obs_000002",
                "obs_000003",
                "obs_000004",
            )
            (run_dir / "run.json").write_text(
                json.dumps({"run_id": "run_test"}),
                encoding="utf-8",
            )
            observations = []
            feature_vectors = (
                [1.0, 0.0],
                [0.99, 0.01],
                [-1.0, 0.0],
                [-1.0, 0.0],
            )
            for observation_id, feature_vector in zip(
                observation_ids, feature_vectors, strict=True
            ):
                screenshot_path = f"screenshots/{observation_id}.png"
                state_path = f"states/{observation_id}.json"
                (run_dir / screenshot_path).write_bytes(b"fixture")
                (run_dir / state_path).write_text("{}", encoding="utf-8")
                observations.append(
                    {
                        "observation_id": observation_id,
                        "screenshot_path": screenshot_path,
                        "view_tree_path": state_path,
                    }
                )
                if observation_id != "obs_000004":
                    torch.save(
                        {
                            "observation_id": observation_id,
                            "patch_features": torch.tensor(
                                [feature_vector], dtype=torch.float32
                            ),
                            "element_logits": torch.logit(torch.tensor([[0.9, 0.9]])),
                            "patch_grid": (1, 1),
                        },
                        feature_dir / f"{observation_id}.pt",
                    )
            (run_dir / "observations.jsonl").write_text(
                "".join(json.dumps(record) + "\n" for record in observations),
                encoding="utf-8",
            )
            (run_dir / "transitions.jsonl").write_text("", encoding="utf-8")
            reference_path = root / "reference.jsonl"
            reference_path.write_text(
                "".join(
                    json.dumps(
                        {
                            "observation_id": observation_id,
                            "cluster_id": "A" if index < 2 else "B",
                            "source": "manual",
                        }
                    )
                    + "\n"
                    for index, observation_id in enumerate(observation_ids)
                ),
                encoding="utf-8",
            )
            deduplication_path = root / "deduplication.jsonl"
            deduplication_groups = (
                ["obs_000001"],
                ["obs_000002"],
                ["obs_000003", "obs_000004"],
            )
            deduplication_path.write_text(
                "".join(
                    json.dumps(
                        {
                            "pixel_sha256": str(index) * 64,
                            "observation_ids": group,
                        }
                    )
                    + "\n"
                    for index, group in enumerate(deduplication_groups, start=1)
                ),
                encoding="utf-8",
            )

            main(
                [
                    str(run_dir),
                    str(feature_dir),
                    str(reference_path),
                    str(deduplication_path),
                    "--output-dir",
                    str(output_root),
                    "--similarity-device",
                    "cpu",
                    "--tile-chunk-size",
                    "1",
                ]
            )

            artifact_dir = output_root / "run_test" / "element_matching_threshold_sweep"
            summary = json.loads(
                (artifact_dir / "summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(1.0, summary["best_full"]["full_pairwise_f1"])
            self.assertEqual(
                (3, 3),
                tuple(
                    torch.load(
                        artifact_dir / "distance_matrix.pt",
                        weights_only=True,
                    ).shape
                ),
            )
            assignments = [
                json.loads(line)
                for line in (artifact_dir / "best_assignments.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertEqual(
                ["obs_000001", "obs_000002", "obs_000003"],
                [record["observation_id"] for record in assignments],
            )
            for filename in (
                "threshold_sweep.csv",
                "threshold_metrics.svg",
                "best_assignments.jsonl",
            ):
                self.assertTrue((artifact_dir / filename).is_file())


if __name__ == "__main__":
    unittest.main()
