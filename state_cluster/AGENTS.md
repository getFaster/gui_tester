# State-Clustering Contributor Guide

This guide covers work inside `state_cluster/`. Follow the repository-root
`AGENTS.md` for shared environment, style, and general contribution rules.
Use `STATE_DATASET.md` as the authoritative end-to-end workflow reference.

## Project Structure & Module Organization

- `state_dataset.py` loads immutable DroidBot runs and validates observation,
  transition, artifact-path, and annotation integrity.
- `state_features.py` creates perceptual, DINO, and grounding feature caches.
- `state_clustering.py` produces automatic cluster annotations.
- `state_deduplicate.py` groups exact decoded screenshots without changing raw
  data or annotations.
- `state_annotation_app.py` provides the Streamlit merge, rename, and outlier
  review interface.
- `state_benchmark.py` and `state_threshold_sweep.py` evaluate cluster
  assignments.
- `state_migrate.py` and `state_migrate_annotation.py` convert legacy files.
- Related tests live in `tests/test_state_*.py`. Canonical workflow data belongs
  under `state_data/<run-id>/`; derived metrics and caches belong under `data/`.

## Build, Test, and Development Commands

Run commands from the repository root:

```powershell
uv run python -m unittest tests.test_state_dataset -v
uv run python -m unittest tests.test_state_pipeline `
  tests.test_state_annotation_merge_ui -v
uv run python -m unittest discover -s tests -v
uv run python -m compileall state_cluster tests
```

Use the first command while changing dataset validation, the second for
annotation behavior, and the full suite plus compilation before handoff.
For workflow CLI examples, including clustering, annotation, benchmarking, and
threshold sweeps, use `STATE_DATASET.md` rather than duplicating commands here.

## State-Workflow Contracts

Treat collector runs as immutable. Annotation records use
`observation_id`, `cluster_id`, and `source`; partial files are valid, but IDs
must be unique and refer to the raw run. Preserve dataset order in generated
records. Keep deduplication independent of annotations and retain singleton
groups. The annotation app must never modify its source annotation and must
resume safely from its output file.

## Testing Guidelines

Use `unittest` and mirror module names, for example
`state_deduplicate.py` → `tests/test_state_deduplicate.py`. Test invalid and
duplicate IDs, partial coverage, deterministic ordering, path safety, and
source-file immutability. UI changes require `streamlit.testing.v1.AppTest`
coverage of real interactions and reruns—not helper-only tests. Tensor or
similarity changes should include shape, dtype, device, and threshold-boundary
cases. Run a representative CLI smoke test when changing a workflow command.
