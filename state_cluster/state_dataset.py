"""Load and validate observation datasets produced by the DroidBot collector."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


ABANDONED_ANNOTATION_FIELDS = frozenset(
    {
        "auto_cluster_id",
        "manual_functional_state_label",
        "manual_substate_label",
    }
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read non-empty JSON objects from an append-only JSONL file."""
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON in {path}:{line_number}") from error
            if not isinstance(value, dict):
                raise ValueError(f"Expected an object in {path}:{line_number}")
            records.append(value)
    return records


@dataclass(frozen=True)
class StateDataset:
    """One immutable collector run with resolved observation artifacts."""

    run_dir: Path
    manifest: dict[str, Any]
    observations: tuple[dict[str, Any], ...]
    transitions: tuple[dict[str, Any], ...]
    observations_by_id: dict[str, dict[str, Any]]

    @classmethod
    def load(cls, run_dir: str | Path, *, validate: bool = True) -> StateDataset:
        run_path = Path(run_dir).resolve()
        manifest_path = run_path / "run.json"
        observations_path = run_path / "observations.jsonl"
        transitions_path = run_path / "transitions.jsonl"
        for required_path in (
            manifest_path,
            observations_path,
            transitions_path,
        ):
            if not required_path.is_file():
                raise FileNotFoundError(f"Missing dataset file: {required_path}")

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        observations = tuple(read_jsonl(observations_path))
        transitions = tuple(read_jsonl(transitions_path))
        observations_by_id: dict[str, dict[str, Any]] = {}
        for observation in observations:
            observation_id = observation.get("observation_id")
            if not isinstance(observation_id, str) or not observation_id:
                raise ValueError("Every observation needs a non-empty observation_id")
            if observation_id in observations_by_id:
                raise ValueError(f"Duplicate observation ID: {observation_id}")
            observations_by_id[observation_id] = observation

        dataset = cls(
            run_dir=run_path,
            manifest=manifest,
            observations=observations,
            transitions=transitions,
            observations_by_id=observations_by_id,
        )
        if validate:
            dataset.validate()
        return dataset

    @property
    def run_id(self) -> str:
        return str(self.manifest["run_id"])

    def resolve_artifact(self, relative_path: str) -> Path:
        """Resolve an artifact while preventing paths from escaping the run."""
        resolved = (self.run_dir / relative_path).resolve()
        if not resolved.is_relative_to(self.run_dir):
            raise ValueError(f"Artifact path escapes run directory: {relative_path}")
        return resolved

    def screenshot_path(self, observation_id: str) -> Path:
        observation = self.observations_by_id[observation_id]
        return self.resolve_artifact(observation["screenshot_path"])

    def state_path(self, observation_id: str) -> Path:
        observation = self.observations_by_id[observation_id]
        return self.resolve_artifact(observation["view_tree_path"])

    def incoming_transitions(
        self, observation_id: str
    ) -> tuple[dict[str, Any], ...]:
        return tuple(
            transition
            for transition in self.transitions
            if transition["destination_observation_id"] == observation_id
        )

    def outgoing_transitions(
        self, observation_id: str
    ) -> tuple[dict[str, Any], ...]:
        return tuple(
            transition
            for transition in self.transitions
            if transition["source_observation_id"] == observation_id
        )

    def group_by(self, field: str) -> dict[Any, list[dict[str, Any]]]:
        groups: dict[Any, list[dict[str, Any]]] = defaultdict(list)
        for observation in self.observations:
            groups[observation.get(field)].append(observation)
        return dict(groups)

    def validate(self) -> None:
        """Check referential integrity, artifact existence, and N+1 continuity."""
        for observation in self.observations:
            observation_id = observation["observation_id"]
            for field in ("screenshot_path", "view_tree_path"):
                relative_path = observation.get(field)
                if not isinstance(relative_path, str):
                    raise ValueError(f"{observation_id} has invalid {field}")
                artifact = self.resolve_artifact(relative_path)
                if not artifact.is_file():
                    raise FileNotFoundError(
                        f"{observation_id} references missing artifact: {artifact}"
                    )

        transition_ids: set[str] = set()
        previous_destination: str | None = None
        for transition in self.transitions:
            transition_id = transition.get("transition_id")
            if not isinstance(transition_id, str) or not transition_id:
                raise ValueError("Every transition needs a transition_id")
            if transition_id in transition_ids:
                raise ValueError(f"Duplicate transition ID: {transition_id}")
            transition_ids.add(transition_id)

            source_id = transition.get("source_observation_id")
            destination_id = transition.get("destination_observation_id")
            if source_id not in self.observations_by_id:
                raise ValueError(
                    f"{transition_id} has unknown source observation: {source_id}"
                )
            if destination_id not in self.observations_by_id:
                raise ValueError(
                    f"{transition_id} has unknown destination observation: "
                    f"{destination_id}"
                )
            if previous_destination is not None and source_id != previous_destination:
                raise ValueError(
                    f"Broken N+1 trajectory at {transition_id}: expected source "
                    f"{previous_destination}, got {source_id}"
                )
            previous_destination = destination_id

        if self.transitions and len(self.observations) != len(self.transitions) + 1:
            raise ValueError(
                "An N+1 run must contain exactly one more observation than "
                "transition"
            )


def load_assignments(paths: Iterable[str | Path]) -> dict[str, dict[str, Any]]:
    """Merge versioned assignment JSONL files by observation ID."""
    assignments: dict[str, dict[str, Any]] = {}
    for path in paths:
        for record in read_jsonl(Path(path)):
            observation_id = record.get("observation_id")
            if not isinstance(observation_id, str):
                raise ValueError(f"Assignment lacks observation_id: {record!r}")
            assignments.setdefault(observation_id, {}).update(record)
    return assignments


def validate_annotations(
    dataset: StateDataset,
    records: Sequence[dict[str, Any]],
    *,
    require_complete: bool = False,
) -> list[dict[str, str]]:
    """Validate and normalize canonical cluster-assignment annotations."""
    annotations: list[dict[str, str]] = []
    seen_ids: set[str] = set()
    for record in records:
        abandoned_fields = ABANDONED_ANNOTATION_FIELDS.intersection(record)
        if abandoned_fields:
            raise ValueError(
                "Abandoned annotation field(s): "
                + ", ".join(sorted(abandoned_fields))
            )
        observation_id = record.get("observation_id")
        if (
            not isinstance(observation_id, str)
            or observation_id not in dataset.observations_by_id
        ):
            raise ValueError(f"Unknown annotated observation: {observation_id}")
        if observation_id in seen_ids:
            raise ValueError(f"Duplicate annotation: {observation_id}")
        cluster_id = record.get("cluster_id")
        if not isinstance(cluster_id, str) or not cluster_id:
            raise ValueError(f"{observation_id} has an invalid cluster_id")
        source = record.get("source")
        if not isinstance(source, str) or not source:
            raise ValueError(f"{observation_id} has an invalid source")
        seen_ids.add(observation_id)
        annotations.append(
            {
                "observation_id": observation_id,
                "cluster_id": cluster_id,
                "source": source,
            }
        )

    if require_complete:
        missing_ids = [
            observation["observation_id"]
            for observation in dataset.observations
            if observation["observation_id"] not in seen_ids
        ]
        if missing_ids:
            raise ValueError(
                "Missing cluster annotations for: " + ", ".join(missing_ids)
            )
    return annotations


def load_annotations(
    dataset: StateDataset,
    path: str | Path,
    *,
    require_complete: bool = False,
) -> list[dict[str, str]]:
    """Load one canonical, possibly partial annotation JSONL file."""
    return validate_annotations(
        dataset,
        read_jsonl(Path(path)),
        require_complete=require_complete,
    )
