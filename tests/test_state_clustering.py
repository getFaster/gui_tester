import json
import tempfile
import unittest
from pathlib import Path

import torch

from state_cluster.state_clustering import (
    distance_matrix_clusters,
    element_matching_distance_matrix,
    main,
    parse_args,
)
from state_cluster.state_dataset import read_jsonl


def make_grounding_payload(
    patch_features: list[list[float]],
    clickable: list[float],
    scrollable: list[float],
) -> dict[str, object]:
    features = torch.tensor(patch_features, dtype=torch.float32)
    probabilities = torch.tensor(
        list(zip(clickable, scrollable, strict=True)),
        dtype=torch.float32,
    )
    return {
        "patch_features": features,
        "element_logits": torch.logit(probabilities),
        "patch_grid": (1, features.shape[0]),
    }


class ElementMatchingTest(unittest.TestCase):
    def test_distance_matrix_is_symmetric_bounded_and_chunk_exact(self) -> None:
        payloads = [
            make_grounding_payload(
                [[1.0, 0.0], [0.0, 1.0]],
                [0.9, 0.2],
                [0.1, 0.8],
            ),
            make_grounding_payload(
                [[0.99, 0.01], [0.1, 0.9]],
                [0.8, 0.3],
                [0.2, 0.7],
            ),
            make_grounding_payload(
                [[-1.0, 0.0]],
                [0.9],
                [0.9],
            ),
        ]

        unchunked = element_matching_distance_matrix(
            payloads,
            device="cpu",
            tile_chunk_size=10,
        )
        chunked = element_matching_distance_matrix(
            payloads,
            device="cpu",
            tile_chunk_size=1,
        )

        torch.testing.assert_close(chunked, unchunked)
        torch.testing.assert_close(chunked, chunked.T)
        torch.testing.assert_close(chunked.diagonal(), torch.zeros(3))
        self.assertTrue(((chunked >= 0) & (chunked <= 1)).all())

    def test_high_confidence_unmatched_tile_increases_distance(self) -> None:
        counterpart = make_grounding_payload(
            [[1.0, 0.0]],
            [0.9],
            [0.9],
        )
        low_confidence_unmatched = make_grounding_payload(
            [[1.0, 0.0], [0.0, 1.0]],
            [0.9, 0.1],
            [0.9, 0.1],
        )
        high_confidence_unmatched = make_grounding_payload(
            [[1.0, 0.0], [0.0, 1.0]],
            [0.9, 0.9],
            [0.9, 0.9],
        )

        low_distance = element_matching_distance_matrix(
            [low_confidence_unmatched, counterpart]
        )[0, 1]
        high_distance = element_matching_distance_matrix(
            [high_confidence_unmatched, counterpart]
        )[0, 1]

        self.assertGreater(high_distance.item(), low_distance.item())

    def test_clickable_and_scrollable_weights_select_separate_scores(self) -> None:
        first = make_grounding_payload(
            [[1.0, 0.0], [0.0, 1.0]],
            [0.9, 0.1],
            [0.1, 0.9],
        )
        second = make_grounding_payload(
            [[1.0, 0.0]],
            [0.9],
            [0.1],
        )

        clickable_distance = element_matching_distance_matrix(
            [first, second],
            class_weights=(1.0, 0.0),
        )[0, 1]
        scrollable_distance = element_matching_distance_matrix(
            [first, second],
            class_weights=(0.0, 1.0),
        )[0, 1]
        equal_distance = element_matching_distance_matrix(
            [first, second],
        )[0, 1]

        self.assertLess(clickable_distance.item(), scrollable_distance.item())
        self.assertAlmostEqual(
            equal_distance.item(),
            ((clickable_distance + scrollable_distance) / 2).item(),
            places=6,
        )

    def test_negative_cosine_match_has_maximum_distance(self) -> None:
        first = make_grounding_payload([[1.0, 0.0]], [0.9], [0.9])
        opposite = make_grounding_payload([[-1.0, 0.0]], [0.9], [0.9])

        distances = element_matching_distance_matrix([first, opposite])

        self.assertEqual(1.0, distances[0, 1].item())

    def test_validates_payload_shapes_and_class_weights(self) -> None:
        malformed = {
            "patch_features": torch.ones(2, 3),
            "element_logits": torch.ones(2, 1),
            "patch_grid": (1, 2),
        }
        with self.assertRaisesRegex(ValueError, "element_logits"):
            element_matching_distance_matrix([malformed])

        payload = make_grounding_payload([[1.0, 0.0]], [0.9], [0.9])
        torch.testing.assert_close(
            element_matching_distance_matrix([payload]),
            torch.zeros(1, 1),
        )
        self.assertEqual(
            [0],
            distance_matrix_clusters(
                torch.zeros(1, 1),
                distance_threshold=0.15,
            ),
        )
        with self.assertRaisesRegex(ValueError, "positive"):
            element_matching_distance_matrix(
                [payload],
                class_weights=(0.0, 0.0),
            )

    def test_precomputed_average_link_clustering(self) -> None:
        labels = distance_matrix_clusters(
            torch.tensor(
                [
                    [0.0, 0.1, 0.8],
                    [0.1, 0.0, 0.8],
                    [0.8, 0.8, 0.0],
                ]
            ),
            distance_threshold=0.2,
        )

        self.assertEqual(labels[0], labels[1])
        self.assertNotEqual(labels[0], labels[2])

    def test_cli_accepts_element_matching_options(self) -> None:
        args = parse_args(
            [
                "run_001",
                "--baseline",
                "element_matching",
                "--feature-dir",
                "features",
                "--clickable-weight",
                "2",
                "--scrollable-weight",
                "1",
                "--similarity-device",
                "cpu",
                "--tile-chunk-size",
                "64",
            ]
        )

        self.assertEqual("element_matching", args.baseline)
        self.assertEqual(2.0, args.clickable_weight)
        self.assertEqual(1.0, args.scrollable_weight)
        self.assertEqual("cpu", args.similarity_device)
        self.assertEqual(64, args.tile_chunk_size)

    def test_element_matching_cli_writes_ordered_assignments(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            run_dir = root / "run"
            feature_dir = root / "features"
            output_dir = root / "clusters"
            (run_dir / "screenshots").mkdir(parents=True)
            (run_dir / "states").mkdir()
            feature_dir.mkdir()
            observation_ids = ("obs_000001", "obs_000002")
            (run_dir / "run.json").write_text(
                json.dumps({"run_id": "run_test"}),
                encoding="utf-8",
            )
            observations = []
            for observation_id in observation_ids:
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
            (run_dir / "observations.jsonl").write_text(
                "".join(json.dumps(record) + "\n" for record in observations),
                encoding="utf-8",
            )
            (run_dir / "transitions.jsonl").write_text("", encoding="utf-8")
            torch.save(
                {
                    "observation_id": observation_ids[0],
                    **make_grounding_payload(
                        [[1.0, 0.0]],
                        [0.9],
                        [0.9],
                    ),
                },
                feature_dir / f"{observation_ids[0]}.pt",
            )
            torch.save(
                {
                    "observation_id": observation_ids[1],
                    **make_grounding_payload(
                        [[0.99, 0.01]],
                        [0.8],
                        [0.8],
                    ),
                },
                feature_dir / f"{observation_ids[1]}.pt",
            )

            main(
                [
                    str(run_dir),
                    "--baseline",
                    "element_matching",
                    "--feature-dir",
                    str(feature_dir),
                    "--output-dir",
                    str(output_dir),
                    "--similarity-device",
                    "cpu",
                    "--tile-chunk-size",
                    "1",
                ]
            )

            records = read_jsonl(
                output_dir
                / "run_test"
                / "annotations"
                / "element_matching.jsonl"
            )
            self.assertEqual(
                list(observation_ids),
                [record["observation_id"] for record in records],
            )
            self.assertTrue(
                all(
                    record["source"] == "element_matching"
                    for record in records
                )
            )
            self.assertTrue(
                all(
                    record["cluster_id"].startswith("element_matching_")
                    for record in records
                )
            )


if __name__ == "__main__":
    unittest.main()
