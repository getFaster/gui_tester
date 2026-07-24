# Repository Guidelines

## Project Structure & Module Organization

This Python 3.13/PyTorch project trains a patch-level GUI element classifier.

- `main.py` is the training entry point for validation, logging, and checkpoints.
- `amex_dataset.py` loads AMEX screenshots and annotations and builds patch masks.
- `dinov3.py` wraps DINOv3 feature extraction.
- `element_finder.py` defines the element-classification model.
- `network.py` contains reusable neural-network layers.
- `screenshot/` and `element_anno/` are expected dataset inputs. `image_data/` contains local sample images.
- `feature/`, `target/`, `runs/`, and `checkpoints/` are generated artifacts and are ignored by Git.

Place new tests in `tests/` and mirror module names, for example `tests/test_amex_dataset.py`.

## Build, Test, and Development Commands

Dependencies are managed by `uv` and pinned in `uv.lock`.

```powershell
uv sync
uv run python main.py --help
uv run python main.py --epochs 1 --batch-size 2
uv run python -m compileall .
```

`uv sync` updates `.venv` with the CUDA 13.2 PyTorch packages. The short training command is a smoke test; populate `screenshot/` and `element_anno/` first. `compileall` catches syntax errors. View logs with `uv run tensorboard --logdir runs`.

## Coding Style & Naming Conventions

Use four-space indentation and standard Python naming: `snake_case` for functions and variables, `PascalCase` for classes, and uppercase names for constants. Add type annotations to public functions and concise docstrings for non-obvious tensor transformations. Keep dimensions explicit, especially `(height, width)`, CHW layout, and row-major patch ordering. Prefer established PyTorch and torchvision operations over custom utilities.

No formatter or linter is configured. Keep imports grouped as standard library, third-party packages, then local modules.

## Testing Guidelines

There is no automated test suite or coverage threshold. New behavior should include focused `pytest` tests when practical. Name tests `test_<behavior>` and validate shapes, dtypes, boundary boxes, and CPU execution. Before a pull request, run syntax checks and a small smoke test.

## Commit & Pull Request Guidelines

History contains only an initial commit, so no commit convention exists. Use short, imperative subjects such as `Add ragged batch validation`. Keep commits focused and exclude datasets, checkpoints, virtual environments, and TensorBoard output.

Pull requests should explain the motivation, summarize behavioral changes, list commands run, and note dataset or CUDA assumptions. Include screenshots or TensorBoard comparisons when changing visible predictions or training behavior, and link relevant issues when available.
