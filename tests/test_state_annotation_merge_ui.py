import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image
from streamlit.testing.v1 import AppTest

from state_dataset import read_jsonl


def run_annotation_app_for_merge_test(
    run_dir: str,
    clusters_path: str,
    reviews_path: str,
    merged_clusters_path: str,
    merged_reviews_path: str,
) -> None:
    """Run the annotation app with isolated merge-test output paths."""
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


def create_two_cluster_fixture(
    root: Path,
    *,
    review_first_cluster: bool = False,
) -> tuple[Path, Path, Path, Path, Path]:
    """Create a two-state run whose states begin in separate clusters."""
    run_dir = root / "run_001"
    screenshots_dir = run_dir / "screenshots"
    states_dir = run_dir / "states"
    screenshots_dir.mkdir(parents=True)
    states_dir.mkdir()

    observations = []
    cluster_records = []
    for index in range(1, 3):
        observation_id = f"obs_{index:06d}"
        screenshot_relative_path = (
            Path("screenshots") / f"{observation_id}.png"
        )
        state_relative_path = Path("states") / f"{observation_id}.json"
        Image.new("RGB", (432, 768), color=(index * 32, 0, 0)).save(
            run_dir / screenshot_relative_path
        )
        (run_dir / state_relative_path).write_text("{}", encoding="utf-8")
        observations.append(
            {
                "observation_id": observation_id,
                "screenshot_path": screenshot_relative_path.as_posix(),
                "view_tree_path": state_relative_path.as_posix(),
                "activity": "org.example/.MainActivity",
            }
        )
        cluster_records.append(
            {
                "observation_id": observation_id,
                "baseline": "test",
                "auto_cluster_id": f"cluster_{index:04d}",
            }
        )

    (run_dir / "run.json").write_text(
        json.dumps({"run_id": "run_001"}),
        encoding="utf-8",
    )
    (run_dir / "observations.jsonl").write_text(
        "".join(
            json.dumps(observation) + "\n" for observation in observations
        ),
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

    clusters_path = root / "clusters.jsonl"
    clusters_path.write_text(
        "".join(json.dumps(record) + "\n" for record in cluster_records),
        encoding="utf-8",
    )
    reviews_path = root / "reviews.jsonl"
    if review_first_cluster:
        reviews_path.write_text(
            json.dumps(
                {
                    "cluster_id": "cluster_0001",
                    "observation_ids": ["obs_000001"],
                    "incorrect_observation_ids": [],
                    "status": "confirmed",
                }
            )
            + "\n",
            encoding="utf-8",
        )
    merged_clusters_path = root / "clusters_merged.jsonl"
    merged_reviews_path = root / "reviews_merged.jsonl"
    return (
        run_dir,
        clusters_path,
        reviews_path,
        merged_clusters_path,
        merged_reviews_path,
    )


def select_and_merge_two_clusters(app: AppTest) -> AppTest:
    """Select both cluster cards and click the merge action."""
    selection_checkboxes = [
        checkbox
        for checkbox in app.checkbox
        if checkbox.label == "Select cluster"
    ]
    if len(selection_checkboxes) != 2:
        raise AssertionError(
            f"Expected two cluster selections, got {len(selection_checkboxes)}"
        )
    selection_checkboxes[0].check().run()

    selection_checkboxes = [
        checkbox
        for checkbox in app.checkbox
        if checkbox.label == "Select cluster"
    ]
    selection_checkboxes[1].check().run()
    merge_button = next(
        button
        for button in app.button
        if button.label == "Merge selected clusters"
    )
    merge_button.click().run()
    return app


class StateAnnotationMergeUiTest(unittest.TestCase):
    def test_merge_clears_instantiated_selection_widgets(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (
                run_dir,
                clusters_path,
                reviews_path,
                merged_clusters_path,
                merged_reviews_path,
            ) = create_two_cluster_fixture(root)

            app = AppTest.from_function(
                run_annotation_app_for_merge_test,
                args=(
                    str(run_dir),
                    str(clusters_path),
                    str(reviews_path),
                    str(merged_clusters_path),
                    str(merged_reviews_path),
                ),
            ).run()
            self.assertEqual([], app.exception)
            select_and_merge_two_clusters(app)

            self.assertEqual([], app.exception)
            self.assertEqual(
                ["cluster_0001", "cluster_0001"],
                [
                    record["auto_cluster_id"]
                    for record in read_jsonl(merged_clusters_path)
                ],
            )

    def test_merge_reviewed_cluster_with_unreviewed_cluster_requires_new_review(
        self,
    ):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (
                run_dir,
                clusters_path,
                reviews_path,
                merged_clusters_path,
                merged_reviews_path,
            ) = create_two_cluster_fixture(
                root,
                review_first_cluster=True,
            )
            original_review = read_jsonl(reviews_path)

            app = AppTest.from_function(
                run_annotation_app_for_merge_test,
                args=(
                    str(run_dir),
                    str(clusters_path),
                    str(reviews_path),
                    str(merged_clusters_path),
                    str(merged_reviews_path),
                ),
            ).run()
            self.assertEqual([], app.exception)
            select_and_merge_two_clusters(app)

            self.assertEqual([], app.exception)
            self.assertEqual(original_review, read_jsonl(reviews_path))
            self.assertFalse(merged_reviews_path.exists())
            self.assertEqual(
                ["cluster_0001", "cluster_0001"],
                [
                    record["auto_cluster_id"]
                    for record in read_jsonl(merged_clusters_path)
                ],
            )

            review_button = next(
                button
                for button in app.button
                if button.label == "Review outliers"
            )
            review_button.click().run()
            self.assertEqual([], app.exception)
            confirm_button = next(
                button
                for button in app.button
                if button.label == "Confirm review"
            )
            confirm_button.click().run()

            self.assertEqual([], app.exception)
            self.assertEqual(original_review, read_jsonl(reviews_path))
            self.assertEqual(
                [
                    {
                        "cluster_id": "cluster_0001",
                        "observation_ids": [
                            "obs_000001",
                            "obs_000002",
                        ],
                        "incorrect_observation_ids": [],
                        "status": "confirmed",
                    }
                ],
                read_jsonl(merged_reviews_path),
            )


if __name__ == "__main__":
    unittest.main()
