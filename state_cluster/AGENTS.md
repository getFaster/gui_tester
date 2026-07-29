## Environment and Commands

Follow `STATE_DATASET.md`. Canonical entry points are:

```powershell
uv run python state_features.py C:\path\to\run_001 --extractor dino
uv run python state_clustering.py C:\path\to\run_001 --baseline structure_str
uv run python state_deduplicate.py C:\path\to\run_001 `
  state_data\run_001\deduplication.jsonl
uv run streamlit run state_annotation_app.py -- `
  --run-dir C:\path\to\run_001 `
  --annotations state_data\run_001\annotations\structure_str.jsonl `
  --deduplication state_data\run_001\deduplication.jsonl `
  --output state_data\run_001\annotations\manual.jsonl
uv run python state_benchmark.py C:\path\to\run_001 `
  state_data\run_001\annotations\structure_str.jsonl `
  state_data\run_001\annotations\manual.jsonl
```

## File and Data Contracts

- `state_dataset.py` loads and validates the immutable DroidBot collector run.
- `state_deduplicate.py` records complete exact-pixel groups without annotation
  data or retained/discarded decisions.
- Annotation JSONL uses only `observation_id`, `cluster_id`, and `source`.
- Partial annotations are valid; IDs must be unique and refer to the raw run.
- The annotation app reviews a fixed representative set and writes only its
  `--output` plus an optional debugging `--reviews` file.
- A completed collector trajectory has one more observation than transition.
  Repeated states and self-transitions remain valid raw data.
