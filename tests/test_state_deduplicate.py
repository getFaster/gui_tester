import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image, PngImagePlugin

from state_dataset import StateDataset, read_jsonl
from state_deduplicate import (
    create_deduplicated_assignments,
    create_deduplicated_run,
    decoded_rgb_sha256,
    evenly_spaced_indices,
    select_cluster_assignments,
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
            (10, 20, 30),
            (10, 20, 30),
            (90, 80, 70),
        ]
        observations = []
        for index, color in enumerate(colors, start=1):
            observation_id = f"obs_{index:06d}"
            screenshot_path = Path("screenshots") / f"{observation_id}.png"
            view_tree_path = Path("states") / f"{observation_id}.json"
            image = Image.new("RGB", (8, 6), color=color)
            if index == 3:
                png_info = PngImagePlugin.PngInfo()
                png_info.add_text("source", "different metadata")
                image.save(
                    self.run_dir / screenshot_path,
                    compress_level=9,
                    pnginfo=png_info,
                )
            else:
                image.save(
                    self.run_dir / screenshot_path,
                    compress_level=index % 10,
                )
            (self.run_dir / view_tree_path).write_text(
                json.dumps({"index": index}),
                encoding="utf-8",
            )
            observations.append(
                {
                    "observation_id": observation_id,
                    "run_id": "run_001",
                    "screenshot_path": screenshot_path.as_posix(),
                    "view_tree_path": view_tree_path.as_posix(),
                }
            )

        transitions = []
        for index in range(1, len(observations)):
            transitions.append(
                {
                    "transition_id": f"trans_{index:06d}",
                    "source_observation_id": observations[index - 1][
                        "observation_id"
                    ],
                    "destination_observation_id": observations[index][
                        "observation_id"
                    ],
                }
            )
        write_jsonl(self.run_dir / "observations.jsonl", observations)
        write_jsonl(self.run_dir / "transitions.jsonl", transitions)
        (self.run_dir / "run.json").write_text(
            json.dumps(
                {
                    "run_id": "run_001",
                    "status": "completed",
                    "observation_count": len(observations),
                    "transition_count": len(transitions),
                }
            ),
            encoding="utf-8",
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_hashes_decoded_pixels_and_dimensions(self):
        first = self.run_dir / "screenshots" / "obs_000001.png"
        metadata_variant = (
            self.run_dir / "screenshots" / "obs_000003.png"
        )
        different = self.run_dir / "screenshots" / "obs_000002.png"
        self.assertNotEqual(first.read_bytes(), metadata_variant.read_bytes())
        self.assertEqual(
            decoded_rgb_sha256(first),
            decoded_rgb_sha256(metadata_variant),
        )
        self.assertNotEqual(
            decoded_rgb_sha256(first),
            decoded_rgb_sha256(different),
        )

        one_pixel_changed = self.root / "one_pixel_changed.png"
        image = Image.new("RGB", (8, 6), color=(10, 20, 30))
        image.putpixel((0, 0), (11, 20, 30))
        image.save(one_pixel_changed)
        self.assertNotEqual(
            decoded_rgb_sha256(first),
            decoded_rgb_sha256(one_pixel_changed),
        )

    def test_evenly_spaced_indices_include_endpoints(self):
        self.assertEqual((0, 2, 5), evenly_spaced_indices(6, 3))
        self.assertEqual((0, 1), evenly_spaced_indices(2, 3))
        self.assertEqual((0,), evenly_spaced_indices(6, 1))
        with self.assertRaisesRegex(ValueError, "at least 1"):
            evenly_spaced_indices(6, 0)

    def test_creates_valid_observation_only_run(self):
        output_dir = self.root / "run_001_deduplicated"
        result = create_deduplicated_run(self.run_dir, output_dir)
        self.assertEqual(output_dir, result)

        derived = StateDataset.load(output_dir)
        retained_ids = [
            observation["observation_id"]
            for observation in derived.observations
        ]
        self.assertEqual(
            [
                "obs_000001",
                "obs_000002",
                "obs_000004",
                "obs_000006",
                "obs_000007",
            ],
            retained_ids,
        )
        self.assertEqual((), derived.transitions)
        self.assertTrue(
            all(
                observation["run_id"] == "run_001_deduplicated"
                for observation in derived.observations
            )
        )

        manifest = derived.manifest
        self.assertEqual("derived", manifest["status"])
        self.assertEqual(5, manifest["observation_count"])
        self.assertEqual(0, manifest["transition_count"])
        self.assertEqual(
            {
                "type": "exact_screenshot_deduplication",
                "source_run_id": "run_001",
                "similarity_metric": "sha256_decoded_rgb_pixels",
                "max_per_image": 3,
                "original_observation_count": 7,
                "retained_observation_count": 5,
                "discarded_observation_count": 2,
            },
            manifest["derivation"],
        )

        audit_records = read_jsonl(output_dir / "deduplication.jsonl")
        first_group = next(
            record
            for record in audit_records
            if "obs_000001" in record["observation_ids"]
        )
        self.assertEqual(
            ["obs_000001", "obs_000004", "obs_000006"],
            first_group["retained_observation_ids"],
        )
        self.assertEqual(
            ["obs_000003", "obs_000005"],
            first_group["discarded_observation_ids"],
        )
        self.assertFalse(
            (output_dir / "screenshots" / "obs_000003.png").exists()
        )
        self.assertFalse(
            (output_dir / "states" / "obs_000005.json").exists()
        )
        for observation_id in retained_ids:
            self.assertTrue(derived.screenshot_path(observation_id).is_file())
            self.assertTrue(derived.state_path(observation_id).is_file())

        source = StateDataset.load(self.run_dir)
        self.assertEqual(7, len(source.observations))
        self.assertEqual(6, len(source.transitions))

    def test_filters_exact_images_only_within_each_cluster(self):
        dataset = StateDataset.load(self.run_dir)
        cluster_ids = ["A", "B", "A", "A", "A", "B", "C"]
        assignments = [
            {
                "observation_id": f"obs_{index:06d}",
                "baseline": "structure_str",
                "auto_cluster_id": cluster_id,
            }
            for index, cluster_id in enumerate(cluster_ids, start=1)
        ]
        selected, audit_records = select_cluster_assignments(
            dataset,
            assignments,
            max_per_image=3,
        )
        self.assertEqual(
            [
                "obs_000001",
                "obs_000002",
                "obs_000003",
                "obs_000005",
                "obs_000006",
                "obs_000007",
            ],
            [
                assignment["observation_id"] for assignment in selected
            ],
        )
        cluster_a_group = next(
            record
            for record in audit_records
            if record["cluster_id"] == "A"
        )
        self.assertEqual(
            ["obs_000004"],
            cluster_a_group["discarded_observation_ids"],
        )

    def test_writes_filtered_assignments_and_audit(self):
        assignments_path = self.root / "assignments.jsonl"
        output_path = self.root / "assignments_deduplicated.jsonl"
        assignments = [
            {
                "observation_id": f"obs_{index:06d}",
                "baseline": "structure_str",
                "auto_cluster_id": "same_cluster",
            }
            for index in (1, 3, 4, 5, 6)
        ]
        write_jsonl(assignments_path, assignments)

        result_path, audit_path = create_deduplicated_assignments(
            self.run_dir,
            assignments_path,
            output_path,
        )
        self.assertEqual(output_path, result_path)
        self.assertEqual(
            self.root / "assignments_deduplicated_deduplication.jsonl",
            audit_path,
        )
        self.assertEqual(
            ["obs_000001", "obs_000004", "obs_000006"],
            [
                record["observation_id"]
                for record in read_jsonl(result_path)
            ],
        )
        audit = read_jsonl(audit_path)
        self.assertEqual(1, len(audit))
        self.assertEqual("same_cluster", audit[0]["cluster_id"])
        self.assertEqual(
            ["obs_000003", "obs_000005"],
            audit[0]["discarded_observation_ids"],
        )
        self.assertEqual(5, len(read_jsonl(assignments_path)))

        with self.assertRaises(FileExistsError):
            create_deduplicated_assignments(
                self.run_dir,
                assignments_path,
                output_path,
            )

    def test_rejects_invalid_or_overlapping_output(self):
        with self.assertRaisesRegex(ValueError, "at least 1"):
            create_deduplicated_run(
                self.run_dir,
                self.root / "invalid",
                max_per_image=0,
            )
        with self.assertRaisesRegex(ValueError, "must not contain"):
            create_deduplicated_run(
                self.run_dir,
                self.run_dir / "derived",
            )

        existing = self.root / "existing"
        existing.mkdir()
        with self.assertRaises(FileExistsError):
            create_deduplicated_run(self.run_dir, existing)


if __name__ == "__main__":
    unittest.main()
