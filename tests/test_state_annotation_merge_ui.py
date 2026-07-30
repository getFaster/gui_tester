import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image
from streamlit.testing.v1 import AppTest

from state_cluster.state_dataset import read_jsonl
from state_cluster.state_deduplicate import create_deduplication_file


def run_annotation_app(
    run_dir: str,
    annotations_path: str,
    deduplication_path: str,
    output_path: str,
    reviews_path: str,
) -> None:
    """Run the canonical annotation workflow in an isolated AppTest."""
    from state_cluster.state_annotation_app import main

    main(
        [
            "--run-dir",
            run_dir,
            "--annotations",
            annotations_path,
            "--deduplication",
            deduplication_path,
            "--output",
            output_path,
            "--reviews",
            reviews_path,
            "--max-per-image",
            "1",
        ]
    )


def create_fixture(
    root: Path,
) -> tuple[Path, Path, Path, Path, Path]:
    run_dir = root / "run_001"
    (run_dir / "screenshots").mkdir(parents=True)
    (run_dir / "states").mkdir()
    observations = []
    colors = ((32, 0, 0), (32, 0, 0), (32, 0, 0))
    for index, color in enumerate(colors, start=1):
        observation_id = f"obs_{index:06d}"
        screenshot_path = Path("screenshots") / f"{observation_id}.png"
        state_path = Path("states") / f"{observation_id}.json"
        Image.new("RGB", (432, 768), color=color).save(
            run_dir / screenshot_path
        )
        (run_dir / state_path).write_text("{}", encoding="utf-8")
        observations.append(
            {
                "observation_id": observation_id,
                "screenshot_path": screenshot_path.as_posix(),
                "view_tree_path": state_path.as_posix(),
            }
        )
    transitions = [
        {
            "transition_id": f"trans_{index:06d}",
            "source_observation_id": f"obs_{index:06d}",
            "destination_observation_id": f"obs_{index + 1:06d}",
        }
        for index in range(1, 3)
    ]
    (run_dir / "run.json").write_text(
        json.dumps({"run_id": "run_001"}), encoding="utf-8"
    )
    for name, records in (
        ("observations.jsonl", observations),
        ("transitions.jsonl", transitions),
    ):
        (run_dir / name).write_text(
            "".join(json.dumps(record) + "\n" for record in records),
            encoding="utf-8",
        )

    annotations_path = root / "structure_str.jsonl"
    source_records = [
        {
            "observation_id": "obs_000001",
            "cluster_id": "cluster_a",
            "source": "structure_str",
        },
        {
            "observation_id": "obs_000002",
            "cluster_id": "cluster_b",
            "source": "structure_str",
        },
        {
            "observation_id": "obs_000003",
            "cluster_id": "cluster_a",
            "source": "structure_str",
        },
    ]
    annotations_path.write_text(
        "".join(json.dumps(record) + "\n" for record in source_records),
        encoding="utf-8",
    )
    deduplication_path = root / "deduplication.jsonl"
    create_deduplication_file(run_dir, deduplication_path)
    output_path = root / "wikipedia.jsonl"
    reviews_path = root / "wikipedia_reviews.jsonl"
    return (
        run_dir,
        annotations_path,
        deduplication_path,
        output_path,
        reviews_path,
    )


def select_and_merge_two_clusters(app: AppTest) -> None:
    checkboxes = [
        checkbox
        for checkbox in app.checkbox
        if checkbox.label == "Select cluster"
    ]
    self_values = [checkbox.value for checkbox in checkboxes]
    if self_values != [False, False]:
        raise AssertionError(self_values)
    checkboxes[0].check().run()
    checkboxes = [
        checkbox
        for checkbox in app.checkbox
        if checkbox.label == "Select cluster"
    ]
    checkboxes[1].check().run()
    next(
        button
        for button in app.button
        if button.label == "Merge selected clusters"
    ).click().run()


class StateAnnotationMergeUiTest(unittest.TestCase):
    def test_deduplicated_merge_reset_and_resume_leave_source_unchanged(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            fixture = create_fixture(Path(temp_dir))
            (
                run_dir,
                annotations_path,
                deduplication_path,
                output_path,
                reviews_path,
            ) = fixture
            source_bytes = annotations_path.read_bytes()
            app = AppTest.from_function(
                run_annotation_app,
                args=tuple(str(path) for path in fixture),
            ).run()
            self.assertEqual([], app.exception)
            self.assertTrue(
                any(
                    "2 representatives from 3 annotated states"
                    in caption.value
                    for caption in app.caption
                )
            )
            self.assertTrue(
                any(
                    "Identical screenshots assigned to different clusters"
                    in warning.value
                    for warning in app.warning
                )
            )

            select_and_merge_two_clusters(app)
            self.assertEqual([], app.exception)
            self.assertEqual(
                ["cluster_a", "cluster_a", "cluster_a"],
                [
                    record["cluster_id"]
                    for record in read_jsonl(output_path)
                ],
            )
            self.assertEqual(
                ["wikipedia"] * 3,
                [record["source"] for record in read_jsonl(output_path)],
            )
            self.assertEqual(source_bytes, annotations_path.read_bytes())

            next(
                button
                for button in app.button
                if button.label == "Reset all"
            ).click().run()
            self.assertEqual([], app.exception)
            self.assertEqual(
                ["cluster_a", "cluster_b", "cluster_a"],
                [
                    record["cluster_id"]
                    for record in read_jsonl(output_path)
                ],
            )

            resumed = AppTest.from_function(
                run_annotation_app,
                args=tuple(str(path) for path in fixture),
            ).run()
            self.assertEqual([], resumed.exception)
            self.assertEqual(2, len(
                [
                    checkbox
                    for checkbox in resumed.checkbox
                    if checkbox.label == "Select cluster"
                ]
            ))

    def test_reviewed_and_unreviewed_merge_requires_current_review(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            fixture = create_fixture(Path(temp_dir))
            reviews_path = fixture[-1]
            reviews_path.write_text(
                json.dumps(
                    {
                        "cluster_id": "cluster_a",
                        "observation_ids": ["obs_000001"],
                        "incorrect_observation_ids": [],
                        "status": "confirmed",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            app = AppTest.from_function(
                run_annotation_app,
                args=tuple(str(path) for path in fixture),
            ).run()
            select_and_merge_two_clusters(app)
            self.assertEqual([], app.exception)

            next(
                button
                for button in app.button
                if button.label == "Review outliers"
            ).click().run()
            next(
                button
                for button in app.button
                if button.label == "Confirm review"
            ).click().run()
            self.assertEqual([], app.exception)
            self.assertEqual(
                [
                    {
                        "cluster_id": "cluster_a",
                        "observation_ids": [
                            "obs_000001",
                            "obs_000002",
                        ],
                        "incorrect_observation_ids": [],
                        "status": "confirmed",
                    }
                ],
                read_jsonl(reviews_path),
            )


if __name__ == "__main__":
    unittest.main()
