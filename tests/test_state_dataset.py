import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from state_dataset import StateDataset, validate_annotations


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


class StateDatasetTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.run_dir = Path(self.temp_dir.name) / "run_001"
        (self.run_dir / "screenshots").mkdir(parents=True)
        (self.run_dir / "states").mkdir()
        (self.run_dir / "run.json").write_text(
            json.dumps({"run_id": "run_001"}),
            encoding="utf-8",
        )
        observations = []
        for index in range(1, 4):
            observation_id = f"obs_{index:06d}"
            screenshot_path = f"screenshots/{observation_id}.png"
            state_path = f"states/{observation_id}.json"
            Image.new("RGB", (432, 768)).save(self.run_dir / screenshot_path)
            (self.run_dir / state_path).write_text("{}", encoding="utf-8")
            observations.append(
                {
                    "observation_id": observation_id,
                    "screenshot_path": screenshot_path,
                    "view_tree_path": state_path,
                    "activity": "org.wikipedia/.MainActivity",
                }
            )
        write_jsonl(self.run_dir / "observations.jsonl", observations)
        write_jsonl(
            self.run_dir / "transitions.jsonl",
            [
                {
                    "transition_id": "trans_000001",
                    "source_observation_id": "obs_000001",
                    "destination_observation_id": "obs_000002",
                    "event": {"event_type": "touch"},
                },
                {
                    "transition_id": "trans_000002",
                    "source_observation_id": "obs_000002",
                    "destination_observation_id": "obs_000003",
                    "event": {"event_type": "key"},
                },
            ],
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_loads_valid_n_plus_one_run(self):
        dataset = StateDataset.load(self.run_dir)
        self.assertEqual("run_001", dataset.run_id)
        self.assertEqual(3, len(dataset.observations))
        self.assertEqual(
            "trans_000001",
            dataset.outgoing_transitions("obs_000001")[0]["transition_id"],
        )
        self.assertEqual(
            "trans_000002",
            dataset.incoming_transitions("obs_000003")[0]["transition_id"],
        )

    def test_rejects_broken_trajectory(self):
        transitions_path = self.run_dir / "transitions.jsonl"
        transitions = [
            {
                "transition_id": "trans_000001",
                "source_observation_id": "obs_000001",
                "destination_observation_id": "obs_000002",
            },
            {
                "transition_id": "trans_000002",
                "source_observation_id": "obs_000001",
                "destination_observation_id": "obs_000003",
            },
        ]
        write_jsonl(transitions_path, transitions)
        with self.assertRaisesRegex(ValueError, "Broken N\\+1 trajectory"):
            StateDataset.load(self.run_dir)

    def test_rejects_artifact_path_outside_run(self):
        observations_path = self.run_dir / "observations.jsonl"
        observations = [
            {
                "observation_id": "obs_000001",
                "screenshot_path": "../outside.png",
                "view_tree_path": "states/obs_000001.json",
            }
        ]
        write_jsonl(observations_path, observations)
        write_jsonl(self.run_dir / "transitions.jsonl", [])
        with self.assertRaisesRegex(ValueError, "escapes run directory"):
            StateDataset.load(self.run_dir)

    def test_validates_partial_canonical_annotations_in_given_order(self):
        dataset = StateDataset.load(self.run_dir)
        records = [
            {
                "observation_id": "obs_000003",
                "cluster_id": "Read_article",
                "source": "structure_str",
            },
            {
                "observation_id": "obs_000001",
                "cluster_id": "Home",
                "source": "structure_str",
            },
        ]
        self.assertEqual(records, validate_annotations(dataset, records))
        with self.assertRaisesRegex(ValueError, "Missing cluster annotations"):
            validate_annotations(dataset, records, require_complete=True)

    def test_rejects_invalid_duplicate_and_abandoned_annotation_records(self):
        dataset = StateDataset.load(self.run_dir)
        valid = {
            "observation_id": "obs_000001",
            "cluster_id": "Home",
            "source": "test",
        }
        cases = (
            ([{**valid, "cluster_id": ""}], "invalid cluster_id"),
            ([valid, valid], "Duplicate annotation"),
            (
                [{**valid, "observation_id": "obs_unknown"}],
                "Unknown annotated observation",
            ),
            (
                [{**valid, "manual_functional_state_label": "Home"}],
                "Abandoned annotation field",
            ),
            (
                [{**valid, "auto_cluster_id": "legacy"}],
                "Abandoned annotation field",
            ),
        )
        for records, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    validate_annotations(dataset, records)


if __name__ == "__main__":
    unittest.main()
