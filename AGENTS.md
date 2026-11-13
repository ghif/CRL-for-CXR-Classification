# Repository Guidelines

## Project Structure

- `prediction/` contains model, dataset, metric, and training implementations. CXR loaders are in `prediction/datasets/`.
- `datafiles/` contains dataset setup utilities and small metadata/configuration files; do not commit private datasets.
- `scripts/` contains shell launchers and experiment entry points. `notebooks/` holds exploratory workflows, and `docs/` records replication and experiment procedures.
- CheXpert-specific baseline scripts and runtime outputs live under `chexpert/`; large images, caches, checkpoints, and generated results should remain local.

## Environment and Common Commands

The main environment is specified by `pyproject.toml` and `poetry.lock` (Python 3.12, PyTorch 2.4.1). Install it with `poetry install`. Useful entry points include:

```bash
poetry run race_invariance --help
poetry run view_invariance --help
python scripts/run_chexpert_table1.py --help
```

For the TPU-specific CheXpert workflow, follow `docs/resnet18-chexpert-tpu-runbook.md` and use `environment-torchtpu.yml`; its dependencies differ from the Poetry environment. Shell scripts may encode cluster-specific paths and arguments, so inspect and adapt them before launching jobs.

## Coding Style

Follow the existing Python style: four-space indentation, `snake_case` for modules/functions/variables, and `PascalCase` for classes. Keep experiment parameters explicit and log seeds, data splits, preprocessing, device, and checkpoint selection. No formatter or linter is configured in `pyproject.toml`; keep diffs focused and run `git diff --check` before committing.

## Testing and Experiment Checks

There is no dedicated automated test suite or coverage target. `scripts/test.sh` launches a training example; it is not a unit-test runner and can consume substantial compute. For code changes, run relevant CLI help or a small, explicitly configured smoke run when appropriate, and report the command and outcome. Do not start full training jobs as a routine check.

## Data, Configuration, and Outputs

Keep `.env`, credentials, patient metadata, images, and generated model artifacts out of version control. Configure dataset locations through the environment variables documented in `docs/replication-guide.md`. Record dataset version/split and any differences from the paper in experiment documentation; never describe defaults as paper settings without source verification.

## Commits and Pull Requests

Recent history uses short, imperative-style subjects (for example, `Add TorchTPU CheXpert ResNet-18 experiment`). Use the same pattern and keep commits focused. A pull request should explain the change, list relevant commands/checks and results, link related issues or experiment reports, and call out data/protocol deviations. Include screenshots only when a visual artifact changes.
