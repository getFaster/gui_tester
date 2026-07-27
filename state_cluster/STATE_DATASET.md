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
```

Tune `--distance-threshold` on a development run before evaluating held-out
runs.

## Annotate and evaluate

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
