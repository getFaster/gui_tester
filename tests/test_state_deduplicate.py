import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image, PngImagePlugin

from state_dataset import StateDataset, read_jsonl
from state_deduplicate import (
    DEDUP_SCOPE_GLOBAL,
    DEDUP_SCOPE_WITHIN_CLUSTER,
    build_deduplication_groups,
    create_deduplication_file,
    decoded_rgb_sha256,
    load_deduplication_groups,
    select_representative_ids,
    validate_deduplication_groups,
)


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


class StateDeduplicateTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.run_dir = self.root / "run_001"
        (self.run_dir / "screenshots").mkdir(parents=True)
        (self.run_dir / "states").mkdir()
        colors = [
            (10, 20, 30),
            (90, 80, 70),
            (10, 20, 30),
            (10, 20, 30),
            (40, 50, 60),
        ]
        observations = []
        for index, color in enumerate(colors, start=1):
            observation_id = f"obs_{index:06d}"
            screenshot_path = Path("screenshots") / f"{observation_id}.png"
            view_tree_path = Path("states") / f"{observation_id}.json"
            image = Image.new("RGB", (8, 6), color=color)
            if index == 3:
                metadata = PngImagePlugin.PngInfo()
                metadata.add_text("source", "different encoding")
                image.save(
                    self.run_dir / screenshot_path,
                    compress_level=9,
                    pnginfo=metadata,
                )
            else:
                image.save(self.run_dir / screenshot_path)
            (self.run_dir / view_tree_path).write_text("{}", encoding="utf-8")
            observations.append(
                {
                    "observation_id": observation_id,
                    "screenshot_path": screenshot_path.as_posix(),
                    "view_tree_path": view_tree_path.as_posix(),
                }
            )
        transitions = [
            {
                "transition_id": f"trans_{index:06d}",
                "source_observation_id": observations[index - 1][
                    "observation_id"
                ],
                "destination_observation_id": observations[index][
                    "observation_id"
                ],
            }
            for index in range(1, len(observations))
        ]
        write_jsonl(self.run_dir / "observations.jsonl", observations)
        write_jsonl(self.run_dir / "transitions.jsonl", transitions)
        (self.run_dir / "run.json").write_text(
            json.dumps({"run_id": "run_001"}), encoding="utf-8"
        )
        self.dataset = StateDataset.load(self.run_dir)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_groups_cover_dataset_deterministically_and_include_singletons(self):
        first = build_deduplication_groups(self.dataset)
        second = build_deduplication_groups(self.dataset)
        self.assertEqual(first, second)
        self.assertEqual(
            [
                ["obs_000001", "obs_000003", "obs_000004"],
                ["obs_000002"],
                ["obs_000005"],
            ],
            [record["observation_ids"] for record in first],
        )
        self.assertEqual(
            [f"obs_{index:06d}" for index in range(1, 6)],
            sorted(
                (
                    observation_id
                    for record in first
                    for observation_id in record["observation_ids"]
                ),
                key=lambda value: int(value.removeprefix("obs_")),
            ),
        )

    def test_hashes_decoded_pixels_not_png_metadata(self):
        first = self.dataset.screenshot_path("obs_000001")
        metadata_variant = self.dataset.screenshot_path("obs_000003")
        self.assertNotEqual(first.read_bytes(), metadata_variant.read_bytes())
        self.assertEqual(
            decoded_rgb_sha256(first), decoded_rgb_sha256(metadata_variant)
        )

    def test_file_refuses_overwrite_without_explicit_option(self):
        output = self.root / "state_data" / "run_001" / "deduplication.jsonl"
        create_deduplication_file(self.run_dir, output)
        self.assertEqual(
            build_deduplication_groups(self.dataset), read_jsonl(output)
        )
        with self.assertRaises(FileExistsError):
            create_deduplication_file(self.run_dir, output)
        create_deduplication_file(self.run_dir, output, overwrite=True)

    def test_rejects_duplicate_missing_unknown_and_out_of_order_ids(self):
        valid = build_deduplication_groups(self.dataset)
        cases = [
            (valid[:-1], "does not cover"),
            (
                valid
                + [
                    {
                        "pixel_sha256": "0" * 64,
                        "observation_ids": ["obs_000001"],
                    }
                ],
                "Duplicate",
            ),
            (
                [
                    {
                        "pixel_sha256": "0" * 64,
                        "observation_ids": ["obs_unknown"],
                    }
                ],
                "Unknown",
            ),
            (
                [
                    {
                        **valid[0],
                        "observation_ids": list(
                            reversed(valid[0]["observation_ids"])
                        ),
                    },
                    *valid[1:],
                ],
                "dataset order",
            ),
        ]
        for records, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    validate_deduplication_groups(self.dataset, records)

    def test_selects_global_and_within_cluster_representatives(self):
        groups = build_deduplication_groups(self.dataset)
        annotations = [
            {
                "observation_id": f"obs_{index:06d}",
                "cluster_id": cluster_id,
                "source": "test",
            }
            for index, cluster_id in enumerate(
                ["A", "B", "A", "C", "B"], start=1
            )
        ]
        self.assertEqual(
            ("obs_000001", "obs_000002", "obs_000005"),
            select_representative_ids(
                self.dataset,
                annotations,
                groups,
                dedup_scope=DEDUP_SCOPE_GLOBAL,
                max_per_image=1,
            ),
        )
        self.assertEqual(
            (
                "obs_000001",
                "obs_000002",
                "obs_000004",
                "obs_000005",
            ),
            select_representative_ids(
                self.dataset,
                annotations,
                groups,
                dedup_scope=DEDUP_SCOPE_WITHIN_CLUSTER,
                max_per_image=1,
            ),
        )

    def test_load_validates_canonical_file(self):
        output = self.root / "deduplication.jsonl"
        create_deduplication_file(self.run_dir, output)
        self.assertEqual(
            build_deduplication_groups(self.dataset),
            load_deduplication_groups(self.dataset, output),
        )


if __name__ == "__main__":
    unittest.main()
