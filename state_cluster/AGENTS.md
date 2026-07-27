## Environment and Commands

For state-dataset work, follow `STATE_DATASET.md`. Typical entry points are:

```powershell
uv run python state_features.py C:\path\to\run_001 --extractor dino
uv run python state_clustering.py C:\path\to\run_001 --baseline state_str
uv run streamlit run state_annotation_app.py -- `
  --run-dir C:\path\to\run_001 `
  --clusters state_clusters\run_001\grounding.jsonl `
  --annotations state_annotations\run_001.jsonl `
  --pairs state_annotations\run_001_pairs.jsonl
uv run python state_benchmark.py --help
```

## File Structure

- `state_dataset.py` loads and validates a DroidBot collector run.
- `state_features.py` extracts perceptual, global DINO, and grounding features
  keyed by `observation_id`.
- `state_clustering.py` builds activity, DroidBot-ID, visual, grounding, and
  transition-aware clustering baselines.
- `state_annotation_app.py` is the Streamlit functional-state annotation GUI.
- `state_benchmark.py` compares cluster assignments with manual labels.
- `STATE_DATASET.md` contains the end-to-end state-dataset commands and schema
  assumptions.


## Data and Model Contracts

- State-dataset files are keyed by `observation_id`. Keep DroidBot
  `state_str`, `structure_str`, automatic cluster IDs, and manual functional
  labels as separate concepts.
- A completed collector trajectory has one more observation than transition.
  Repeated states and ineffective/self-transitions are valid data and must not
  be discarded.
