import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from state_dataset import StateDataset, read_jsonl
from state_migrate import (
    convert_legacy_annotation,
    migrate_legacy_annotation_in_place,
)


class StateMigrationTest(unittest.TestCase):
    def test_conversion_preserves_order_and_rejects_overwrite(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            run_dir = root / "run_001"
            (run_dir / "screenshots").mkdir(parents=True)
            (run_dir / "states").mkdir()
            observations = []
            for index in range(1, 3):
                observation_id = f"obs_{index:06d}"
                screenshot = f"screenshots/{observation_id}.png"
                state = f"states/{observation_id}.json"
                Image.new("RGB", (8, 6)).save(run_dir / screenshot)
                (run_dir / state).write_text("{}", encoding="utf-8")
                observations.append(
                    {
                        "observation_id": observation_id,
                        "screenshot_path": screenshot,
                        "view_tree_path": state,
                    }
                )
            (run_dir / "run.json").write_text(
                json.dumps({"run_id": "run_001"}), encoding="utf-8"
            )
            (run_dir / "observations.jsonl").write_text(
                "".join(json.dumps(record) + "\n" for record in observations),
                encoding="utf-8",
            )
            (run_dir / "transitions.jsonl").write_text(
                json.dumps(
                    {
                        "transition_id": "trans_000001",
                        "source_observation_id": "obs_000001",
                        "destination_observation_id": "obs_000002",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            legacy = root / "legacy.jsonl"
            legacy.write_text(
                "".join(
                    json.dumps(record) + "\n"
                    for record in (
                        {
                            "observation_id": "obs_000002",
                            "auto_cluster_id": "B",
                            "baseline": "old",
                        },
                        {
                            "observation_id": "obs_000001",
                            "auto_cluster_id": "A",
                            "baseline": "old",
                        },
                    )
                ),
                encoding="utf-8",
            )
            output = root / "state_data" / "annotations" / "manual.jsonl"
            convert_legacy_annotation(
                StateDataset.load(run_dir),
                legacy,
                output,
                source="manual",
            )
            self.assertEqual(
                [
                    {
                        "observation_id": "obs_000002",
                        "cluster_id": "B",
                        "source": "manual",
                    },
                    {
                        "observation_id": "obs_000001",
                        "cluster_id": "A",
                        "source": "manual",
                    },
                ],
                read_jsonl(output),
            )
            with self.assertRaises(FileExistsError):
                convert_legacy_annotation(
                    StateDataset.load(run_dir),
                    legacy,
                    output,
                    source="manual",
                )

            backup = migrate_legacy_annotation_in_place(
                StateDataset.load(run_dir),
                legacy,
            )
            self.assertEqual(root / "legacy.legacy.jsonl", backup)
            self.assertEqual(
                [
                    {
                        "observation_id": "obs_000002",
                        "cluster_id": "B",
                        "source": "old",
                    },
                    {
                        "observation_id": "obs_000001",
                        "cluster_id": "A",
                        "source": "old",
                    },
                ],
                read_jsonl(legacy),
            )
            self.assertEqual(
                [
                    {
                        "observation_id": "obs_000002",
                        "auto_cluster_id": "B",
                        "baseline": "old",
                    },
                    {
                        "observation_id": "obs_000001",
                        "auto_cluster_id": "A",
                        "baseline": "old",
                    },
                ],
                read_jsonl(backup),
            )
            with self.assertRaises(FileExistsError):
                migrate_legacy_annotation_in_place(
                    StateDataset.load(run_dir),
                    legacy,
                )


if __name__ == "__main__":
    unittest.main()
