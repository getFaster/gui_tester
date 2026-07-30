# DroidBot State Dataset Workflow

The canonical workflow is:

```text
Collect immutable dataset
→ generate deduplication information once
→ generate an automatic annotation
→ manually edit a representative view
→ benchmark against the corrected annotation
```

## Canonical files

```text
droidbot/runs/dataset/run_006/       # immutable collector output

state_data/run_006/
    deduplication.jsonl
    annotations/
        structure_str.jsonl
        wikipedia.jsonl
    debug/
        wikipedia_reviews.jsonl
```

`state_cluster/` contains Python source only. `state_data/` contains canonical
workflow data. Feature caches, checkpoints, and metrics are derived artifacts
under ignored `data/`. The old `state_clusters/` and `state_annotations/`
directories are deprecated and should remain untouched until migrated data is
verified.

Every annotation record uses one schema:

```json
{"observation_id":"obs_000001","cluster_id":"Read_article","source":"structure_str"}
```

Annotation files may be partial. Observation IDs must exist in the raw run and
may occur only once. `cluster_id` and `source` must be non-empty strings.
Legacy automatic IDs, manual-label fields, and pairwise files are rejected.

## Generate exact-screenshot groups

Deduplication depends only on decoded RGB pixels and dimensions. It writes one
group for every screenshot identity, including singletons, and never copies or
changes the dataset:

```powershell
uv run .\state_cluster\state_deduplicate.py `
  .\droidbot\runs\dataset\run_006 `
  .\state_data\run_006\deduplication.jsonl
```

An existing file is protected. Use `--overwrite` only when intentionally
regenerating it from the same immutable run.

## Generate an automatic annotation

```powershell
uv run .\state_cluster\state_clustering.py `
  .\droidbot\runs\dataset\run_006 `
  --baseline structure_str
```

The default output is
`state_data/run_006/annotations/structure_str.jsonl`. Learned baselines still
accept `--feature-dir`; feature caches remain derived data.

## Edit a fixed representative view

```powershell
uv run streamlit run .\state_cluster\state_annotation_app.py -- `
  --run-dir .\droidbot\runs\dataset\run_006 `
  --annotations .\state_data\run_006\annotations\structure_str.jsonl `
  --deduplication .\state_data\run_006\deduplication.jsonl `
  --output .\state_data\run_006\annotations\wikipedia.jsonl `
  --reviews .\state_data\run_006\debug\wikipedia_reviews.jsonl `
  --dedup-scope within-cluster `
  --max-per-image 3
```

The app never writes the source annotation or raw run. It resumes an existing
`--output`. Representative observation IDs are selected once from the source
annotation, so merges do not change which observations are under review.
Merges and renames apply to the complete output annotation; outlier decisions
apply to the selected observations.

`within-cluster` is the default and preserves representation for every source
cluster. It retains identical screenshots separately when their source cluster
IDs differ. Use `--dedup-scope global` to select without cluster membership.
The optional review log is debugging information and is not consumed by other
workflow stages.

## Benchmark the corrected annotation

```powershell
uv run .\state_cluster\state_benchmark.py `
  .\droidbot\runs\dataset\run_006 `
  .\state_data\run_006\annotations\structure_str.jsonl `
  .\state_data\run_006\annotations\wikipedia.jsonl
```

Only observations present in both partial annotation files are evaluated. The
second file is the manually corrected reference.

## Tune the element-matching threshold

First extract grounding features only for the first observation in every exact
screenshot group:

```powershell
uv run .\state_cluster\state_features.py `
  .\droidbot\runs\dataset\run_006 `
  --extractor grounding `
  --deduplication .\state_data\run_006\deduplication.jsonl
```

The exhaustive sweep clusters those representatives, computes the
element-matching distance matrix once, builds one complete average-linkage
tree, and evaluates every threshold that produces a distinct clustering.

```powershell
uv run .\state_cluster\state_threshold_sweep.py `
  .\droidbot\runs\dataset\run_006 `
  .\state_features\run_006\grounding `
  .\state_data\run_006\annotations\final_annotation.jsonl `
  .\state_data\run_006\deduplication.jsonl
```

The default artifact directory is
`data/run_006/element_matching_threshold_sweep/`. It contains:

- `threshold_sweep.csv`: pairwise F1, ARI, and NMI for every effective
  threshold over the deduplicated representatives;
- `summary.json`: the current `0.15` result and the best thresholds selected by
  pairwise F1, then ARI, then NMI;
- `threshold_metrics.svg`: the full-view metric curves;
- `best_assignments.jsonl`: assignments at the best full-view threshold; and
- `distance_matrix.pt`: the computed screen-distance matrix.

The selected value is a benchmark-specific optimum when the same manual
annotation is used for threshold selection and reporting. Freeze it before
evaluating generalization on another independently annotated run or app.

## Migration

Migration creates new files and leaves old directories in place:

```text
state_clusters/run_006/structure_str.jsonl
  → state_data/run_006/annotations/structure_str.jsonl

state_clusters/run_006/structure_str_merged.jsonl
  → state_data/run_006/annotations/wikipedia.jsonl
```

Convert `auto_cluster_id` to `cluster_id`, convert `baseline` to `source`, and
preserve record order and cluster IDs. Generate `deduplication.jsonl` from the
raw run with the command above. Keep useful review logs only under
`state_data/run_006/debug/`.

## Verification

```powershell
$env:PYTHONUTF8 = "1"
$env:PYTHONPATH = "state_cluster;element_finder"
uv run python -m unittest discover -s tests -v
uv run python -m compileall state_cluster tests
```
