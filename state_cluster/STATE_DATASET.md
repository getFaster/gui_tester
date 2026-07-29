# DroidBot State Dataset Workflow

## Collect a run

Collection uses DroidBot's existing fixed event interval. It does not poll or
resize screenshots.

```powershell
python -m droidbot.start `
  -a C:\path\to\app-prod-debug.apk `
  -d emulator-5554 `
  -o C:\path\to\output `
  -policy dfs_greedy `
  -interval 1 `
  -count 100 `
  --collect-dataset `
  --dataset-run-id run_001
```

The run is stored below `output/dataset/run_001`. A completed trajectory has
one more observation than transition. Repeated DroidBot hashes still receive
new observation IDs, and self-transitions remain in `transitions.jsonl`.

## Validate and extract features

All commands preserve the native screenshot files. The 432 by 768 emulator
size is divisible by DINOv3's 16-pixel patch size.

```powershell
uv run python state_features.py C:\path\to\run_001 --extractor perceptual
uv run python state_features.py C:\path\to\run_001 --extractor dino
$env:PYTHONUTF8 = "1"  # required by PyTorch Inductor on CP950 Windows hosts
uv run python state_features.py C:\path\to\run_001 --extractor grounding `
  --grounding-checkpoint checkpoints\element_finder\best.pt
```

## Deduplicate exact screenshots

Create a separate observation-only dataset that keeps at most three evenly
spaced occurrences of each exact screenshot:

```powershell
uv run python state_deduplicate.py `
  C:\path\to\run_001 `
  C:\path\to\run_001_deduplicated `
  --max-per-image 3
```

Screenshot identity is based on decoded RGB pixels and image dimensions, so
PNG compression and metadata do not affect the result. The source run is not
modified. The derived run has no transitions because globally removing
observations cannot preserve the original event trajectory. Its
`deduplication.jsonl` records every repeated-image group and which observation
IDs were retained or discarded. Regenerate features, clusters, and annotations
for the derived run instead of reusing outputs keyed to the source run.

To retain an existing clustering and its cluster IDs, filter an assignment
file instead of copying the dataset:

```powershell
uv run python state_deduplicate.py `
  C:\path\to\run_006 `
  state_clusters\run_006\structure_str_merged_deduplicated.jsonl `
  --assignments state_clusters\run_006\structure_str_merged.jsonl `
  --max-per-image 3
```

This mode compares screenshots only among observations with the same
`auto_cluster_id`. It writes a subset of the original assignment records and a
neighboring `_deduplication.jsonl` audit file. The original dataset, cluster
IDs, and clustering file are unchanged. Because the output intentionally omits
some observations, use it as a representative assignment subset for downstream
analysis rather than as input to the annotation UI, which requires complete
dataset coverage.

## Build clustering baselines

Categorical baselines do not require feature files:

```powershell
uv run python state_clustering.py C:\path\to\run_001 --baseline activity
uv run python state_clustering.py C:\path\to\run_001 --baseline state_str
uv run python state_clustering.py C:\path\to\run_001 --baseline structure_str
```

Visual and learned baselines use the matching extractor directory:

```powershell
uv run python state_clustering.py C:\path\to\run_001 --baseline perceptual `
  --feature-dir state_features\run_001
uv run python state_clustering.py C:\path\to\run_001 --baseline dino `
  --feature-dir state_features\run_001\dino_global
uv run python state_clustering.py C:\path\to\run_001 --baseline grounding `
  --feature-dir state_features\run_001\grounding
uv run python state_clustering.py C:\path\to\run_001 `
  --baseline grounding_transition `
  --feature-dir state_features\run_001\grounding
uv run python state_clustering.py C:\path\to\run_001 `
  --baseline element_matching `
  --feature-dir state_features\run_001\grounding
```

`element_matching` compares all contextual ElementFinder tile embeddings
between each pair of screens. It computes clickable and scrollable scores
separately, weights every tile pair by both sigmoid probabilities, and uses a
bidirectional best match so an important element missing from either screen
reduces similarity. The default class weights are equal. Use
`--clickable-weight` and `--scrollable-weight` to change them,
`--similarity-device cpu` to force CPU execution, or `--tile-chunk-size` to
limit the temporary tile-pair matrix. Chunking remains an exact all-pairs
comparison.

Tune `--distance-threshold` on a development run before evaluating held-out
runs. The initial `0.15` default is shared with the other embedding baselines
and should be included in a threshold sweep rather than treated as calibrated
for element matching.

## Annotate and evaluate

The annotation interface accepts partial assignment files by default. It
displays the retained and total observation counts and only reviews assigned
observations against the original run. Add `--strict-assignments` to require an
assignment for every observation. Assignment files are JSON Lines regardless
of whether their filename ends in `.jsonl` or `.json`.

Start the local annotation interface with one baseline assignment:

```powershell
uv run streamlit run state_annotation_app.py -- `
  --run-dir C:\path\to\run_001 `
  --clusters state_clusters\run_001\grounding.jsonl `
  --annotations state_annotations\run_001.jsonl `
  --pairs state_annotations\run_001_pairs.jsonl
```

Assigning the same functional label across clusters merges them in the manual
ground truth. Selecting only part of a cluster and assigning another label
splits it. Invalid observations are excluded from evaluation.

```powershell
uv run python state_benchmark.py `
  C:\path\to\run_001 `
  state_clusters\run_001\grounding.jsonl `
  state_annotations\run_001.jsonl `
  --output state_clusters\run_001\grounding_metrics.json
```

Add `--include-substate` to evaluate the finer functional-state plus scroll
substate labeling.
