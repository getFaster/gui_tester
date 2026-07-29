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
    cluster_count: int = 2,
) -> tuple[Path, Path, Path, Path, Path]:
    """Create a run whose states begin in separate clusters."""
    run_dir = root / "run_001"
    screenshots_dir = run_dir / "screenshots"
    states_dir = run_dir / "states"
    screenshots_dir.mkdir(parents=True)
    states_dir.mkdir()

    observations = []
    cluster_records = []
    for index in range(1, cluster_count + 1):
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
        "".join(
            json.dumps(
                {
                    "transition_id": f"trans_{index:06d}",
                    "source_observation_id": f"obs_{index:06d}",
                    "destination_observation_id": f"obs_{index + 1:06d}",
                }
            )
            + "\n"
            for index in range(1, cluster_count)
        ),
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
    def test_outlier_review_can_rename_current_cluster(self):
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

            new_name_input = next(
                text_input
                for text_input in app.text_input
                if text_input.label == "New name for this cluster"
            )
            new_name_input.input("home").run()
            rename_button = next(
                button
                for button in app.button
                if button.label == "Rename cluster"
            )
            rename_button.click().run()

            self.assertEqual([], app.exception)
            self.assertEqual(
                {"home", "cluster_0002"},
                {
                    record["auto_cluster_id"]
                    for record in read_jsonl(merged_clusters_path)
                },
            )
            cluster_picker = next(
                selectbox
                for selectbox in app.selectbox
                if selectbox.label == "Cluster to review"
            )
            self.assertEqual("home", cluster_picker.value)
            self.assertTrue(
                any(
                    option.startswith("home ·")
                    for option in cluster_picker.options
                )
            )
            self.assertFalse(
                any(
                    option.startswith("cluster_0001 ·")
                    for option in cluster_picker.options
                )
            )
            self.assertTrue(cluster_picker.options[0].startswith("home ·"))
            self.assertTrue(
                cluster_picker.options[1].startswith("cluster_0002 ·")
            )

    def test_rename_cluster_updates_outlier_review_and_review_record(self):
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

            cluster_to_rename = next(
                selectbox
                for selectbox in app.selectbox
                if selectbox.label == "Cluster to rename"
            )
            cluster_to_rename.set_value("cluster_0001").run()
            new_name_input = next(
                text_input
                for text_input in app.text_input
                if text_input.label == "New cluster name"
            )
            new_name_input.input("home").run()
            rename_button = next(
                button for button in app.button if button.label == "Rename"
            )
            rename_button.click().run()

            self.assertEqual([], app.exception)
            self.assertEqual(
                {"home", "cluster_0002"},
                {
                    record["auto_cluster_id"]
                    for record in read_jsonl(merged_clusters_path)
                },
            )
            cluster_picker = next(
                selectbox
                for selectbox in app.selectbox
                if selectbox.label == "Cluster to review"
            )
            self.assertTrue(
                any(option.startswith("home ·") for option in cluster_picker.options)
            )
            self.assertFalse(
                any(
                    option.startswith("cluster_0001 ·")
                    for option in cluster_picker.options
                )
            )
            self.assertEqual(
                [
                    {
                        "cluster_id": "home",
                        "observation_ids": ["obs_000001"],
                        "incorrect_observation_ids": [],
                        "status": "confirmed",
                    }
                ],
                read_jsonl(merged_reviews_path),
            )

    def test_outlier_review_reflects_merged_cluster_membership(self):
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

            cluster_picker = next(
                selectbox
                for selectbox in app.selectbox
                if selectbox.label == "Cluster to review"
            )
            self.assertEqual("cluster_0001", cluster_picker.value)
            self.assertEqual(1, len(cluster_picker.options))
            self.assertIn("2 state(s)", cluster_picker.options[0])
            current_review_checkboxes = [
                checkbox
                for checkbox in app.checkbox
                if checkbox.label == "Does not belong"
                and "incorrect::current::" in str(checkbox.key)
            ]
            self.assertEqual(2, len(current_review_checkboxes))
            self.assertTrue(
                any(
                    "obs_000001" in str(checkbox.key)
                    for checkbox in current_review_checkboxes
                )
            )
            self.assertTrue(
                any(
                    "obs_000002" in str(checkbox.key)
                    for checkbox in current_review_checkboxes
                )
            )

            confirm_button = next(
                button
                for button in app.button
                if button.label == "Confirm and continue"
            )
            confirm_button.click().run()
            self.assertFalse(reviews_path.exists())
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

    def test_merged_review_can_group_selected_outliers_together(self):
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

            review_button = next(
                button
                for button in app.button
                if button.label == "Review outliers"
            )
            review_button.click().run()
            self.assertEqual([], app.exception)

            merged_outlier_checkboxes = [
                checkbox
                for checkbox in app.checkbox
                if checkbox.label == "Does not belong"
                and "incorrect::merged::" in str(checkbox.key)
            ]
            self.assertEqual(2, len(merged_outlier_checkboxes))
            merged_outlier_checkboxes[0].check().run()
            merged_outlier_checkboxes = [
                checkbox
                for checkbox in app.checkbox
                if checkbox.label == "Does not belong"
                and "incorrect::merged::" in str(checkbox.key)
            ]
            merged_outlier_checkboxes[1].check().run()

            merged_grouping_radio = next(
                radio
                for radio in app.radio
                if radio.label
                == "How should the selected outliers be clustered?"
                and "outlier-grouping::merged::" in str(radio.key)
            )
            merged_grouping_radio.set_value("together").run()
            confirm_button = next(
                button
                for button in app.button
                if button.label == "Confirm review"
            )
            confirm_button.click().run()

            self.assertEqual([], app.exception)
            updated_assignments = read_jsonl(merged_clusters_path)
            cluster_ids = {
                record["auto_cluster_id"] for record in updated_assignments
            }
            self.assertEqual(1, len(cluster_ids))
            self.assertTrue(next(iter(cluster_ids)).startswith("outlier_group_"))

    def test_unselect_all_clears_cluster_selections(self):
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

            selection_checkboxes = [
                checkbox
                for checkbox in app.checkbox
                if checkbox.label == "Select cluster"
            ]
            selection_checkboxes[0].check().run()
            selection_checkboxes = [
                checkbox
                for checkbox in app.checkbox
                if checkbox.label == "Select cluster"
            ]
            selection_checkboxes[1].check().run()

            unselect_button = next(
                button
                for button in app.button
                if button.label == "Unselect all"
            )
            unselect_button.click().run()

            self.assertEqual([], app.exception)
            self.assertTrue(
                all(
                    not checkbox.value
                    for checkbox in app.checkbox
                    if checkbox.label == "Select cluster"
                )
            )
            merge_button = next(
                button
                for button in app.button
                if button.label == "Merge selected clusters"
            )
            self.assertTrue(merge_button.disabled)

    def test_selection_persists_across_cluster_pages(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (
                run_dir,
                clusters_path,
                reviews_path,
                merged_clusters_path,
                merged_reviews_path,
            ) = create_two_cluster_fixture(root, cluster_count=13)

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

            page_size = next(
                selectbox
                for selectbox in app.selectbox
                if selectbox.label == "Clusters per page"
            )
            page_size.select(12).run()
            first_selection = next(
                checkbox
                for checkbox in app.checkbox
                if checkbox.label == "Select cluster"
            )
            first_selection.check().run()

            page_number = next(
                number_input
                for number_input in app.number_input
                if number_input.label == "Page"
            )
            page_number.set_value(2).run()
            self.assertEqual([], app.exception)
            self.assertTrue(
                any(
                    caption.value.endswith("1 selected")
                    for caption in app.caption
                )
            )

            page_number = next(
                number_input
                for number_input in app.number_input
                if number_input.label == "Page"
            )
            page_number.set_value(1).run()
            first_selection = next(
                checkbox
                for checkbox in app.checkbox
                if checkbox.label == "Select cluster"
            )
            self.assertTrue(first_selection.value)

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
