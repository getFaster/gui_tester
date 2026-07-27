"""Local Streamlit interface for functional-state annotation."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Sequence

from state_dataset import StateDataset, read_jsonl


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


def save_annotations(
    path: Path,
    existing: dict[str, dict[str, Any]],
    observation_ids: Sequence[str],
    *,
    functional_label: str,
    substate_label: str,
    status: str,
) -> None:
    for observation_id in observation_ids:
        existing[observation_id] = {
            "observation_id": observation_id,
            "manual_functional_state_label": functional_label or None,
            "manual_substate_label": substate_label or None,
            "status": status,
        }
    _atomic_write_records(path, existing)


def save_pair_decision(
    path: Path,
    existing: dict[str, dict[str, Any]],
    first_id: str,
    second_id: str,
    decision: str,
) -> None:
    ordered_ids = sorted((first_id, second_id))
    pair_id = "__".join(ordered_ids)
    existing[pair_id] = {
        "pair_id": pair_id,
        "first_observation_id": ordered_ids[0],
        "second_observation_id": ordered_ids[1],
        "decision": decision,
    }
    _atomic_write_records(path, existing)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--clusters", type=Path, required=True)
    parser.add_argument(
        "--annotations",
        type=Path,
        default=Path("state_annotations/manual_annotations.jsonl"),
    )
    parser.add_argument(
        "--pairs",
        type=Path,
        default=Path("state_annotations/manual_pairs.jsonl"),
    )
    args, _ = parser.parse_known_args(argv)
    return args


def _render_observation(st: Any, dataset: StateDataset, observation_id: str) -> None:
    observation = dataset.observations_by_id[observation_id]
    st.image(
        str(dataset.screenshot_path(observation_id)),
        caption=observation_id,
        use_container_width=True,
    )
    st.caption(
        f"{observation.get('activity')} · "
        f"state={observation.get('droidbot_state_str')} · "
        f"structure={observation.get('droidbot_structure_str')}"
    )


def main(argv: Sequence[str] | None = None) -> None:
    import streamlit as st

    args = parse_args(argv)
    dataset = StateDataset.load(args.run_dir)
    assignments = read_jsonl(args.clusters)
    cluster_by_observation = {
        record["observation_id"]: record["auto_cluster_id"]
        for record in assignments
    }
    observations_by_cluster: dict[str, list[str]] = {}
    for observation in dataset.observations:
        observation_id = observation["observation_id"]
        cluster_id = cluster_by_observation.get(observation_id, "unassigned")
        observations_by_cluster.setdefault(cluster_id, []).append(observation_id)

    annotations = _records_by_key(args.annotations, "observation_id")
    pair_decisions = _records_by_key(args.pairs, "pair_id")

    st.set_page_config(page_title="Functional State Annotation", layout="wide")
    st.title("Functional State Annotation")
    st.caption(
        f"{dataset.run_id}: {len(dataset.observations)} observations, "
        f"{len(dataset.transitions)} transitions"
    )

    cluster_ids = sorted(observations_by_cluster)
    selected_cluster = st.sidebar.selectbox("Automatic cluster", cluster_ids)
    cluster_observation_ids = observations_by_cluster[selected_cluster]
    st.sidebar.metric("Cluster size", len(cluster_observation_ids))

    cluster_tab, pair_tab, transition_tab = st.tabs(
        ("Cluster correction", "Pair comparison", "Transitions")
    )
    with cluster_tab:
        selected_ids = st.multiselect(
            "Observations to update",
            cluster_observation_ids,
            default=cluster_observation_ids,
        )
        columns = st.columns(min(4, max(1, len(cluster_observation_ids))))
        for index, observation_id in enumerate(cluster_observation_ids):
            with columns[index % len(columns)]:
                _render_observation(st, dataset, observation_id)
                current = annotations.get(observation_id)
                if current:
                    st.json(current)

        functional_label = st.text_input("Functional-state label")
        substate_label = st.text_input("Substate label")
        status = st.selectbox(
            "Status",
            ("valid", "transient", "invalid", "uncertain"),
        )
        if st.button("Save selected observations", type="primary"):
            if not selected_ids:
                st.error("Select at least one observation.")
            else:
                save_annotations(
                    args.annotations,
                    annotations,
                    selected_ids,
                    functional_label=functional_label.strip(),
                    substate_label=substate_label.strip(),
                    status=status,
                )
                st.success(
                    f"Saved {len(selected_ids)} observation annotations. "
                    "Assign the same label across clusters to merge them; select "
                    "a subset to split a cluster."
                )

    all_observation_ids = [
        observation["observation_id"] for observation in dataset.observations
    ]
    with pair_tab:
        first_id = st.selectbox("First observation", all_observation_ids)
        second_candidates = [
            observation_id
            for observation_id in all_observation_ids
            if observation_id != first_id
        ]
        second_id = st.selectbox("Second observation", second_candidates)
        first_column, second_column = st.columns(2)
        with first_column:
            _render_observation(st, dataset, first_id)
            st.json(
                {
                    "incoming": dataset.incoming_transitions(first_id),
                    "outgoing": dataset.outgoing_transitions(first_id),
                }
            )
        with second_column:
            _render_observation(st, dataset, second_id)
            st.json(
                {
                    "incoming": dataset.incoming_transitions(second_id),
                    "outgoing": dataset.outgoing_transitions(second_id),
                }
            )
        decision = st.radio(
            "Same functional state?",
            ("same", "different", "uncertain"),
            horizontal=True,
        )
        if st.button("Save pair decision"):
            save_pair_decision(
                args.pairs,
                pair_decisions,
                first_id,
                second_id,
                decision,
            )
            st.success("Pair decision saved.")

    with transition_tab:
        transition_observation_id = st.selectbox(
            "Observation",
            all_observation_ids,
            key="transition_observation",
        )
        screenshot_column, metadata_column = st.columns((1, 2))
        with screenshot_column:
            _render_observation(st, dataset, transition_observation_id)
        with metadata_column:
            st.subheader("Incoming")
            st.json(dataset.incoming_transitions(transition_observation_id))
            st.subheader("Outgoing")
            st.json(dataset.outgoing_transitions(transition_observation_id))
            st.subheader("Device state")
            st.json(
                json.loads(
                    dataset.state_path(transition_observation_id).read_text(
                        encoding="utf-8"
                    )
                )
            )


if __name__ == "__main__":
    main()
