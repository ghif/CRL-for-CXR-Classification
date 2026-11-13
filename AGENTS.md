# Repository Guidelines

## Project Structure

- `prediction/` contains models, metrics, training, and CXR loaders; data utilities are in `datafiles/`.
- `scripts/` contains experiment entry points, `notebooks/` exploratory work, and `docs/` replication guides and reports.
- `chexpert/` holds CheXpert-specific scripts and ignored runtime outputs. Keep images, caches, checkpoints, and generated results local.

## Environment and Common Commands

The general environment is pinned in `pyproject.toml` and `poetry.lock` (Python 3.12, PyTorch 2.4.1). Install and inspect entry points with:

```bash
poetry run race_invariance --help
poetry run view_invariance --help
python scripts/run_chexpert_table1.py --help
```

For TPU CheXpert runs, use `environment-torchtpu.yml` and follow `docs/resnet18-chexpert-tpu-runbook.md`; this environment differs from Poetry. Review cluster-specific shell scripts before use.

## TorchXRayVision TPU Inference

- Compile fixed-shape inference with `torch.compile(model, backend="tpu", dynamic=False)`. The first call compiles; benchmark later calls with the same shape. On this host, eager batch-32 was about 5 images/s; compiled batch-64 evaluated roughly 27k examples in about a minute.
- Benchmark batch sizes. Pad the final partial batch and trim its padded scores to avoid compiling another shape. Apply XRV's grayscale normalization to `[-1024, 1024]`; skip crop/resize only when inputs are already square 224 × 224.
- XRV operating-point calibration uses indexed assignment unsupported by this TorchTPU compiler. For AUROC, raw logits preserve ranking; keep calibration for threshold-based metrics.

## Coding Style

Use four-space indentation, `snake_case` for functions and variables, and `PascalCase` for classes. Log seeds, splits, preprocessing, device, and checkpoint selection. No formatter/linter is configured; keep diffs focused and run `git diff --check`.

## Testing and Experiment Checks

There is no dedicated automated test suite or coverage target. `scripts/test.sh` launches training and may consume substantial compute; it is not a unit-test runner. Prefer relevant CLI help or a small configured smoke run, and report commands and outcomes. Avoid full training jobs for routine checks.

## Data, Configuration, and Outputs

Keep `.env`, credentials, patient metadata, images, and model artifacts out of Git. Configure data paths as documented in `docs/replication-guide.md`. Record dataset version/split and paper deviations; verify paper settings against the source.

## Commits and Pull Requests

Use short imperative commit subjects (for example, `Add TorchTPU CheXpert ResNet-18 experiment`) and focused commits. PRs should explain changes, list checks/results, link issues or reports, and note data/protocol deviations. Add screenshots when visual artifacts change.
