"""Review whether observations belong in their automatically assigned clusters."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Sequence

from state_dataset import StateDataset, read_jsonl


REVIEW_IMAGE_WIDTH = 320


def _records_by_key(path: Path, key: str) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    return {str(record[key]): record for record in read_jsonl(path)}


def _atomic_write_records(
    path: Path,
    records: dict[str, dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(path.name + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as output:
        for record_key in sorted(records):
            output.write(
                json.dumps(records[record_key], ensure_ascii=False) + "\n"
            )
    os.replace(temporary_path, path)


def build_cluster_groups(
    dataset: StateDataset,
    assignments: Sequence[dict[str, Any]],
) -> dict[str, tuple[str, ...]]:
    """Validate assignments and group observations in dataset order."""
    cluster_by_observation: dict[str, str] = {}
    for record in assignments:
        observation_id = record.get("observation_id")
        cluster_id = record.get("auto_cluster_id")
        if observation_id not in dataset.observations_by_id:
            raise ValueError(f"Unknown assigned observation: {observation_id}")
        if not isinstance(cluster_id, str) or not cluster_id:
            raise ValueError(
                f"{observation_id} has an invalid auto_cluster_id"
            )
        if observation_id in cluster_by_observation:
            raise ValueError(f"Duplicate assignment: {observation_id}")
        cluster_by_observation[observation_id] = cluster_id

    missing_ids = [
        observation["observation_id"]
        for observation in dataset.observations
        if observation["observation_id"] not in cluster_by_observation
    ]
    if missing_ids:
        raise ValueError(
            "Missing cluster assignments for: " + ", ".join(missing_ids)
        )

    grouped: dict[str, list[str]] = {}
    for observation in dataset.observations:
        observation_id = observation["observation_id"]
        cluster_id = cluster_by_observation[observation_id]
        grouped.setdefault(cluster_id, []).append(observation_id)
    return {
        cluster_id: tuple(observation_ids)
        for cluster_id, observation_ids in grouped.items()
    }


def ordered_cluster_ids(
    groups: dict[str, tuple[str, ...]],
) -> list[str]:
    """Put larger comparison groups first and singleton groups last."""
    return sorted(
        groups,
        key=lambda cluster_id: (
            len(groups[cluster_id]) == 1,
            cluster_id,
        ),
    )


def review_is_current(
    review: dict[str, Any] | None,
    observation_ids: Sequence[str],
) -> bool:
    """Return whether a saved review matches the current cluster membership."""
    return bool(
        review
        and review.get("status") == "confirmed"
        and review.get("observation_ids") == list(observation_ids)
    )


def save_cluster_review(
    path: Path,
    existing: dict[str, dict[str, Any]],
    cluster_id: str,
    observation_ids: Sequence[str],
    incorrect_observation_ids: Sequence[str],
) -> None:
    """Persist one confirmed cluster review, replacing an older decision."""
    observation_id_set = set(observation_ids)
    incorrect_id_set = set(incorrect_observation_ids)
    if len(observation_id_set) != len(observation_ids):
        raise ValueError("A cluster contains duplicate observation IDs")
    if not incorrect_id_set.issubset(observation_id_set):
        raise ValueError("Incorrect observations must belong to the cluster")

    existing[cluster_id] = {
        "cluster_id": cluster_id,
        "observation_ids": list(observation_ids),
        "incorrect_observation_ids": [
            observation_id
            for observation_id in observation_ids
            if observation_id in incorrect_id_set
        ],
        "status": "confirmed",
    }
    _atomic_write_records(path, existing)


def write_cluster_assignments(
    path: Path,
    assignments: Sequence[dict[str, Any]],
) -> None:
    """Atomically write assignments using the clustering pipeline schema."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(path.name + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as output:
        for record in assignments:
            observation_id = record.get("observation_id")
            baseline = record.get("baseline")
            cluster_id = record.get("auto_cluster_id")
            if not isinstance(observation_id, str) or not observation_id:
                raise ValueError("Every assignment needs an observation_id")
            if not isinstance(baseline, str) or not baseline:
                raise ValueError(
                    f"{observation_id} has an invalid baseline"
                )
            if not isinstance(cluster_id, str) or not cluster_id:
                raise ValueError(
                    f"{observation_id} has an invalid auto_cluster_id"
                )
            output.write(
                json.dumps(
                    {
                        "observation_id": observation_id,
                        "baseline": baseline,
                        "auto_cluster_id": cluster_id,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    os.replace(temporary_path, path)


def merge_cluster_assignments(
    assignments: Sequence[dict[str, Any]],
    selected_cluster_ids: Sequence[str],
) -> tuple[list[dict[str, Any]], str]:
    """Merge selected clusters under their lexicographically smallest ID."""
    selected_ids = set(selected_cluster_ids)
    if len(selected_ids) < 2:
        raise ValueError("Select at least two different clusters to merge")

    existing_ids = {
        str(record.get("auto_cluster_id")) for record in assignments
    }
    missing_ids = sorted(selected_ids - existing_ids)
    if missing_ids:
        raise ValueError(
            "Unknown clusters selected for merging: " + ", ".join(missing_ids)
        )

    merged_cluster_id = min(selected_ids)
    merged_assignments: list[dict[str, Any]] = []
    for record in assignments:
        updated = dict(record)
        if updated.get("auto_cluster_id") in selected_ids:
            updated["auto_cluster_id"] = merged_cluster_id
        merged_assignments.append(updated)
    return merged_assignments, merged_cluster_id


def representative_observation_ids(
    observation_ids: Sequence[str],
    *,
    maximum: int = 3,
) -> tuple[str, ...]:
    """Choose evenly distributed screenshots for a compact cluster preview."""
    if maximum < 1:
        raise ValueError("maximum must be positive")
    if len(observation_ids) <= maximum:
        return tuple(observation_ids)
    if maximum == 1:
        return (observation_ids[0],)
    indices = {
        round(index * (len(observation_ids) - 1) / (maximum - 1))
        for index in range(maximum)
    }
    return tuple(observation_ids[index] for index in sorted(indices))


def flagged_observations(
    cluster_ids: Sequence[str],
    groups: dict[str, tuple[str, ...]],
    reviews: dict[str, dict[str, Any]],
) -> dict[str, tuple[str, ...]]:
    """Collect flagged observations from current, confirmed reviews."""
    flagged: dict[str, tuple[str, ...]] = {}
    for cluster_id in cluster_ids:
        review = reviews.get(cluster_id)
        if not review_is_current(review, groups[cluster_id]):
            continue
        incorrect_ids = tuple(review.get("incorrect_observation_ids", ()))
        if incorrect_ids:
            flagged[cluster_id] = incorrect_ids
    return flagged


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--clusters", type=Path, required=True)
    parser.add_argument(
        "--reviews",
        type=Path,
        help=(
            "Review JSONL output. Defaults to "
            "state_annotations/<run>_<cluster-file>_reviews.jsonl."
        ),
    )
    parser.add_argument(
        "--merged-clusters",
        type=Path,
        help=(
            "Merged assignment JSONL output. Defaults to a _merged file beside "
            "the original cluster assignments."
        ),
    )
    parser.add_argument(
        "--merged-reviews",
        type=Path,
        help=(
            "Reviews created from the cluster-merging gallery. Defaults to a "
            "_merged file beside the normal review output."
        ),
    )
    args, _ = parser.parse_known_args(argv)
    return args


def _default_reviews_path(args: argparse.Namespace, run_id: str) -> Path:
    if args.reviews is not None:
        return args.reviews
    return Path("state_annotations") / (
        f"{run_id}_{args.clusters.stem}_reviews.jsonl"
    )


def _default_merged_clusters_path(args: argparse.Namespace) -> Path:
    if args.merged_clusters is not None:
        return args.merged_clusters
    return args.clusters.with_name(f"{args.clusters.stem}_merged.jsonl")


def _default_merged_reviews_path(
    args: argparse.Namespace,
    reviews_path: Path,
) -> Path:
    if args.merged_reviews is not None:
        return args.merged_reviews
    return reviews_path.with_name(
        f"{reviews_path.stem}_merged{reviews_path.suffix}"
    )


def _review_widget_scope(
    mode: str,
    reviews_path: Path,
    cluster_id: str,
    observation_ids: Sequence[str],
) -> str:
    membership = "\0".join(observation_ids).encode("utf-8")
    membership_hash = hashlib.sha256(membership).hexdigest()[:12]
    return (
        f"{mode}::{reviews_path.resolve()}::{cluster_id}::{membership_hash}"
    )


def _incorrect_widget_key(
    review_scope: str,
    cluster_id: str,
    observation_id: str,
) -> str:
    return f"incorrect::{review_scope}::{cluster_id}::{observation_id}"


def _clear_review_widget_state(
    st: Any,
    review_scope: str,
    cluster_id: str,
    observation_ids: Sequence[str],
) -> None:
    for observation_id in observation_ids:
        st.session_state.pop(
            _incorrect_widget_key(
                review_scope,
                cluster_id,
                observation_id,
            ),
            None,
        )


def _render_observation(
    st: Any,
    dataset: StateDataset,
    observation_id: str,
    *,
    selectable: bool,
    cluster_id: str,
    review_scope: str,
    selected_by_default: bool = False,
) -> bool:
    observation = dataset.observations_by_id[observation_id]
    with st.container(border=True):
        st.image(
            str(dataset.screenshot_path(observation_id)),
            width=REVIEW_IMAGE_WIDTH,
        )
        selected = False
        if selectable:
            selected = st.checkbox(
                "Does not belong",
                value=selected_by_default,
                key=_incorrect_widget_key(
                    review_scope,
                    cluster_id,
                    observation_id,
                ),
            )
        st.caption(f"**{observation_id}**")
        activity = observation.get("activity")
        if activity:
            st.caption(str(activity))
    return selected


def _render_flagged_queue(
    st: Any,
    dataset: StateDataset,
    flagged: dict[str, tuple[str, ...]],
) -> None:
    st.subheader("States to revisit")
    if not flagged:
        st.success("No states were marked as incorrect.")
        return

    total = sum(len(observation_ids) for observation_ids in flagged.values())
    st.info(
        f"{total} state{'s' if total != 1 else ''} from "
        f"{len(flagged)} cluster{'s' if len(flagged) != 1 else ''} "
        "need follow-up."
    )
    for cluster_id, observation_ids in flagged.items():
        st.markdown(f"#### {cluster_id}")
        columns = st.columns(min(4, len(observation_ids)))
        for index, observation_id in enumerate(observation_ids):
            with columns[index % len(columns)]:
                _render_observation(
                    st,
                    dataset,
                    observation_id,
                    selectable=False,
                    cluster_id=cluster_id,
                    review_scope="flagged",
                )


def _render_outlier_review(
    st: Any,
    dataset: StateDataset,
    groups: dict[str, tuple[str, ...]],
    reviews_path: Path,
    reviews: dict[str, dict[str, Any]],
) -> None:
    cluster_ids = ordered_cluster_ids(groups)
    reviewed_ids = [
        cluster_id
        for cluster_id in cluster_ids
        if review_is_current(reviews.get(cluster_id), groups[cluster_id])
    ]
    pending_ids = [
        cluster_id
        for cluster_id in cluster_ids
        if cluster_id not in reviewed_ids
    ]
    flagged = flagged_observations(cluster_ids, groups, reviews)
    flagged_count = sum(len(ids) for ids in flagged.values())
    total_clusters = len(cluster_ids)
    completed_clusters = len(reviewed_ids)
    progress = completed_clusters / total_clusters if total_clusters else 1.0

    st.caption(
        "Select states that do not belong in the displayed cluster, then "
        "confirm to continue."
    )
    st.progress(
        progress,
        text=f"{completed_clusters} of {total_clusters} clusters reviewed",
    )
    metric_columns = st.columns(3)
    metric_columns[0].metric("Remaining clusters", len(pending_ids))
    metric_columns[1].metric("Flagged states", flagged_count)
    metric_columns[2].metric("Reviewed clusters", completed_clusters)

    if not cluster_ids:
        st.warning("The assignment file contains no clusters.")
        return

    if not pending_ids:
        st.success("All clusters have been reviewed.")
        _render_flagged_queue(st, dataset, flagged)
        return

    cluster_id = pending_ids[0]
    observation_ids = groups[cluster_id]
    review_scope = _review_widget_scope(
        "original",
        reviews_path,
        cluster_id,
        observation_ids,
    )
    review_number = completed_clusters + 1
    group_kind = (
        "Multi-state cluster" if len(observation_ids) > 1 else "Singleton cluster"
    )
    st.subheader(f"{group_kind} {review_number} of {total_clusters}")
    st.caption(f"{cluster_id} · {len(observation_ids)} state(s)")

    columns = st.columns(min(4, len(observation_ids)))
    incorrect_ids: list[str] = []
    for index, observation_id in enumerate(observation_ids):
        with columns[index % len(columns)]:
            if _render_observation(
                st,
                dataset,
                observation_id,
                selectable=True,
                cluster_id=cluster_id,
                review_scope=review_scope,
            ):
                incorrect_ids.append(observation_id)

    if len(observation_ids) == 1:
        st.caption(
            "For a singleton, select the state if this cluster should be "
            "reconsidered or merged later."
        )
    elif incorrect_ids:
        st.caption(
            f"{len(incorrect_ids)} state(s) will be added to the follow-up queue."
        )
    else:
        st.caption(
            "Nothing selected: every state will be accepted in this cluster."
        )

    if st.button(
        "Confirm and continue",
        type="primary",
        use_container_width=True,
    ):
        save_cluster_review(
            reviews_path,
            reviews,
            cluster_id,
            observation_ids,
            incorrect_ids,
        )
        st.rerun()


def _selection_key(merged_path: Path, cluster_id: str) -> str:
    return f"merge-select::{merged_path.resolve()}::{cluster_id}"


def _render_cluster_card(
    st: Any,
    dataset: StateDataset,
    merged_path: Path,
    cluster_id: str,
    observation_ids: Sequence[str],
) -> bool:
    with st.container(border=True):
        st.markdown(f"**{cluster_id}**")
        st.caption(
            f"{len(observation_ids)} "
            f"state{'s' if len(observation_ids) != 1 else ''}"
        )
        preview_ids = representative_observation_ids(observation_ids)
        st.image(
            [
                str(dataset.screenshot_path(observation_id))
                for observation_id in preview_ids
            ],
            width=92,
        )
        st.checkbox(
            "Select cluster",
            key=_selection_key(merged_path, cluster_id),
        )
        return st.button(
            "Review outliers",
            key=f"merge-review::{merged_path.resolve()}::{cluster_id}",
            use_container_width=True,
        )


def _focused_review_key(merged_path: Path) -> str:
    return f"focused-review::{merged_path.resolve()}"


def _render_focused_cluster_review(
    st: Any,
    dataset: StateDataset,
    merged_path: Path,
    cluster_id: str,
    observation_ids: Sequence[str],
    reviews_path: Path,
    reviews: dict[str, dict[str, Any]],
) -> None:
    st.subheader(f"Review {cluster_id}")
    st.caption(
        f"{len(observation_ids)} current state(s) · Select every state that "
        "does not belong in this cluster."
    )

    saved_review = reviews.get(cluster_id)
    saved_review_is_current = review_is_current(
        saved_review,
        observation_ids,
    )
    saved_incorrect_ids = (
        set(saved_review.get("incorrect_observation_ids", ()))
        if saved_review_is_current and saved_review is not None
        else set()
    )
    if saved_review_is_current:
        st.info("This cluster has a saved review. You can update it below.")

    review_scope = _review_widget_scope(
        "merged",
        reviews_path,
        cluster_id,
        observation_ids,
    )
    columns = st.columns(min(4, len(observation_ids)))
    incorrect_ids: list[str] = []
    for index, observation_id in enumerate(observation_ids):
        with columns[index % len(columns)]:
            if _render_observation(
                st,
                dataset,
                observation_id,
                selectable=True,
                cluster_id=cluster_id,
                review_scope=review_scope,
                selected_by_default=observation_id in saved_incorrect_ids,
            ):
                incorrect_ids.append(observation_id)

    action_columns = st.columns((3, 1))
    if action_columns[0].button(
        "Confirm review",
        type="primary",
        use_container_width=True,
    ):
        save_cluster_review(
            reviews_path,
            reviews,
            cluster_id,
            observation_ids,
            incorrect_ids,
        )
        st.session_state.pop(_focused_review_key(merged_path), None)
        st.toast(f"Saved review for {cluster_id}")
        st.rerun()

    if action_columns[1].button(
        "Back to gallery",
        use_container_width=True,
    ):
        _clear_review_widget_state(
            st,
            review_scope,
            cluster_id,
            observation_ids,
        )
        st.session_state.pop(_focused_review_key(merged_path), None)
        st.rerun()

    st.caption(f"Merged-cluster reviews: {reviews_path}")


def _render_merge_preview(
    st: Any,
    dataset: StateDataset,
    groups: dict[str, tuple[str, ...]],
    selected_cluster_ids: Sequence[str],
) -> None:
    with st.expander("Preview selected clusters", expanded=True):
        columns = st.columns(min(3, len(selected_cluster_ids)))
        for index, cluster_id in enumerate(selected_cluster_ids):
            with columns[index % len(columns)]:
                st.markdown(f"**{cluster_id}**")
                observation_ids = groups[cluster_id]
                preview_ids = representative_observation_ids(
                    observation_ids,
                    maximum=4,
                )
                st.image(
                    [
                        str(dataset.screenshot_path(observation_id))
                        for observation_id in preview_ids
                    ],
                    width=150,
                )
                st.caption(f"{len(observation_ids)} state(s)")


def _render_merge_tab(
    st: Any,
    dataset: StateDataset,
    original_assignments: Sequence[dict[str, Any]],
    merged_assignments: Sequence[dict[str, Any]],
    merged_path: Path,
    merged_reviews_path: Path,
    merged_reviews: dict[str, dict[str, Any]],
) -> None:
    groups = build_cluster_groups(dataset, merged_assignments)
    cluster_ids = ordered_cluster_ids(groups)
    original_cluster_count = len(
        build_cluster_groups(dataset, original_assignments)
    )
    merged_cluster_count = len(cluster_ids)

    st.caption(
        "Select clusters that represent the same functional state. The "
        "original assignment file is never modified."
    )
    focused_key = _focused_review_key(merged_path)
    focused_cluster_id = st.session_state.get(focused_key)
    if focused_cluster_id not in groups:
        st.session_state.pop(focused_key, None)
        focused_cluster_id = None
    if focused_cluster_id is not None:
        _render_focused_cluster_review(
            st,
            dataset,
            merged_path,
            focused_cluster_id,
            groups[focused_cluster_id],
            merged_reviews_path,
            merged_reviews,
        )
        return

    metric_columns = st.columns(3)
    metric_columns[0].metric("Current clusters", merged_cluster_count)
    metric_columns[1].metric(
        "Clusters merged",
        original_cluster_count - merged_cluster_count,
    )
    metric_columns[2].metric("States", len(dataset.observations))

    control_columns = st.columns((2, 1, 1))
    filter_text = control_columns[0].text_input(
        "Filter by cluster ID",
        placeholder="Type part of a cluster ID",
    )
    page_size = control_columns[1].selectbox(
        "Clusters per page",
        (12, 24, 48),
        index=1,
    )
    filtered_ids = [
        cluster_id
        for cluster_id in cluster_ids
        if filter_text.lower() in cluster_id.lower()
    ]
    page_count = max(1, (len(filtered_ids) + page_size - 1) // page_size)
    page_number = control_columns[2].number_input(
        "Page",
        min_value=1,
        max_value=page_count,
        value=1,
        step=1,
    )
    page_start = (int(page_number) - 1) * page_size
    page_ids = filtered_ids[page_start : page_start + page_size]

    selected_ids = [
        cluster_id
        for cluster_id in cluster_ids
        if st.session_state.get(_selection_key(merged_path, cluster_id), False)
    ]
    st.caption(
        f"Showing {len(page_ids)} of {len(filtered_ids)} matching clusters · "
        f"{len(selected_ids)} selected"
    )

    if not page_ids:
        st.info("No cluster IDs match the filter.")
    else:
        gallery_columns = st.columns(4)
        requested_review_id = None
        for index, cluster_id in enumerate(page_ids):
            with gallery_columns[index % len(gallery_columns)]:
                if _render_cluster_card(
                    st,
                    dataset,
                    merged_path,
                    cluster_id,
                    groups[cluster_id],
                ):
                    requested_review_id = cluster_id
        if requested_review_id is not None:
            st.session_state[focused_key] = requested_review_id
            st.rerun()

    selected_ids = [
        cluster_id
        for cluster_id in cluster_ids
        if st.session_state.get(_selection_key(merged_path, cluster_id), False)
    ]
    if selected_ids:
        _render_merge_preview(st, dataset, groups, selected_ids)

    action_columns = st.columns((3, 1, 1))
    if action_columns[0].button(
        "Merge selected clusters",
        type="primary",
        disabled=len(selected_ids) < 2,
        use_container_width=True,
    ):
        updated_assignments, merged_cluster_id = merge_cluster_assignments(
            merged_assignments,
            selected_ids,
        )
        history_key = f"merge-history::{merged_path.resolve()}"
        history = st.session_state.setdefault(history_key, [])
        history.append([dict(record) for record in merged_assignments])
        write_cluster_assignments(merged_path, updated_assignments)
        for cluster_id in selected_ids:
            st.session_state[_selection_key(merged_path, cluster_id)] = False
        st.toast(f"Merged into {merged_cluster_id}")
        st.rerun()

    history_key = f"merge-history::{merged_path.resolve()}"
    history = st.session_state.setdefault(history_key, [])
    if action_columns[1].button(
        "Undo last merge",
        disabled=not history,
        use_container_width=True,
    ):
        previous_assignments = history.pop()
        write_cluster_assignments(merged_path, previous_assignments)
        st.rerun()

    reset_requested = action_columns[2].button(
        "Reset all",
        disabled=merged_cluster_count == original_cluster_count,
        use_container_width=True,
    )
    if reset_requested:
        history.append([dict(record) for record in merged_assignments])
        write_cluster_assignments(merged_path, original_assignments)
        for cluster_id in cluster_ids:
            st.session_state[_selection_key(merged_path, cluster_id)] = False
        st.rerun()

    st.caption(f"Merged assignments: {merged_path}")
    if history:
        st.caption("Undo history is available for this browser session.")


def main(argv: Sequence[str] | None = None) -> None:
    import streamlit as st

    args = parse_args(argv)
    dataset = StateDataset.load(args.run_dir)
    original_assignments = read_jsonl(args.clusters)
    original_groups = build_cluster_groups(dataset, original_assignments)
    reviews_path = _default_reviews_path(args, dataset.run_id)
    reviews = _records_by_key(reviews_path, "cluster_id")
    merged_reviews_path = _default_merged_reviews_path(args, reviews_path)
    if merged_reviews_path.resolve() == reviews_path.resolve():
        raise ValueError(
            "Merged-cluster reviews must not overwrite original reviews"
        )
    merged_reviews = _records_by_key(merged_reviews_path, "cluster_id")
    merged_path = _default_merged_clusters_path(args)
    if merged_path.resolve() == args.clusters.resolve():
        raise ValueError("Merged output must not overwrite the input assignments")
    merged_assignments = (
        read_jsonl(merged_path)
        if merged_path.is_file()
        else original_assignments
    )
    build_cluster_groups(dataset, merged_assignments)

    st.set_page_config(
        page_title="Cluster Verification",
        page_icon="✓",
        layout="wide",
    )
    st.title("Cluster Verification")
    st.caption(f"{dataset.run_id} · {len(dataset.observations)} states")

    outlier_tab, merge_tab = st.tabs(("Outlier review", "Cluster merging"))
    with outlier_tab:
        _render_outlier_review(
            st,
            dataset,
            original_groups,
            reviews_path,
            reviews,
        )
    with merge_tab:
        _render_merge_tab(
            st,
            dataset,
            original_assignments,
            merged_assignments,
            merged_path,
            merged_reviews_path,
            merged_reviews,
        )


if __name__ == "__main__":
    main()
