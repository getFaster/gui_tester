# Repository Guidelines

## Project Scope

This Python 3.13/PyTorch repository has two related workflows:

1. Train and run a patch-level GUI element classifier on AMEX screenshots.
2. Extract features from DroidBot observation runs, cluster UI states, annotate
   functional states, and benchmark automatic state abstractions.

The DroidBot crawler and collector live in the separate `droidbot` repository.
This repository consumes completed collector run directories; do not add
crawler behavior or Android-device control here.

## Project Structure

- `element_finder/` contains the element-classification model and training code.
- `tests/` contains the focused `unittest` suite. Mirror module names when
  adding tests, for example `tests/test_state_dataset.py`.
- `state_cluster/` contains the state-dataset feature extraction, clustering,
  annotation, and benchmarking code.
- `data/` is the ignored subdirectory for screenshots, annotations, extracted
  features, checkpoints, TensorBoard logs, and state-dataset artifacts.

## Environment and Commands

Dependencies are managed by `uv` and pinned in `uv.lock`. Use the project
environment for all runtime checks because system Python may not contain the
required CUDA and FlexAttention packages.

```powershell
uv sync
uv run python -m unittest discover -s tests -v
uv run python -m compileall .
```

On Windows hosts using a CP950 code page, set `$env:PYTHONUTF8 = "1"` before
grounding-feature extraction if PyTorch Inductor otherwise fails while reading
UTF-8 templates.

## Data and Model Contracts

- Make image size around 432 by 768 pixels. Collector runs currently use 432 by 768 screenshots.
- Images are CHW tensors. Always state spatial dimensions as `(height, width)`.
- DINOv3 uses 16 by 16 patches.

## Coding Style

Use four-space indentation and standard Python naming: `snake_case` for
functions and variables, `PascalCase` for classes, and uppercase names for
constants. Add type annotations to public functions and concise docstrings for
non-obvious tensor or graph transformations.

Black is the formatter. Group imports as standard library, third-party
packages, then local modules. Keep changes focused and preserve unrelated user
work in a dirty worktree.

## Testing

Use `unittest`; `pytest` is not a project dependency. Add focused tests for new
behavior and run the smallest relevant test module during iteration, followed
by the full suite before handoff:

```powershell
uv run python -m unittest tests.test_state_dataset tests.test_state_pipeline -v
uv run python -m unittest discover -s tests -v
```

For tensor code, test shapes, dtypes, boundary boxes, height/width order,
flattened patch alignment, and CPU execution where practical. For state data,
test referential integrity, path safety, repeated observations,
self-transitions, and invalid/manual-label handling. Run a representative CLI
smoke test for workflow changes; syntax checks alone are insufficient.
