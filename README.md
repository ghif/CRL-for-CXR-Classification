# Causal Representation Learning with Observational Grouping for CXR Classification

This repository contains code for chest X-ray (CXR) classification experiments involving observational grouping, protected attributes, and representation invariance, along with a conventional ResNet-18 CheXpert baseline. The baseline described below is a completed, single-seed adapted reproduction of the CheXpert tasks in the paper; it is not the full multi-seed Table 1 replication.

## Repository layout

| Path | Contents |
|---|---|
| `prediction/` | Models, training entry points, metrics, and CXR datasets/loaders |
| `prediction/datasets/` | Dataset-specific loading and preprocessing logic |
| `datafiles/` | Data setup utilities and small metadata files; large/private data is not tracked |
| `chexpert/` | Hugging Face extraction/preparation and ResNet-18 baseline runner |
| `scripts/` | Shell launchers and experiment scripts |
| `notebooks/` | Exploratory and analysis notebooks |
| `docs/` | Replication plans, runbook, and detailed experiment report |

## Environment

The general project environment is defined by `pyproject.toml` and `poetry.lock` (Python 3.12, PyTorch 2.4.1). Install and inspect the project commands with:

```bash
poetry install
poetry run race_invariance --help
poetry run view_invariance --help
```

The ResNet-18 run documented here used the conda environment `crl-cxr` with TorchTPU. Its dependencies and versions differ from the Poetry environment; use [`environment-torchtpu.yml`](environment-torchtpu.yml) and the [TPU runbook](docs/resnet18-chexpert-tpu-runbook.md) when recreating that setup. The ResNet runner explicitly selects TPU by default and fails if the TPU runtime is unavailable.

## ResNet-18 CheXpert baseline

### Tasks and reported results

The experiment trains separate binary classifiers for **No Finding vs Pleural Effusion** and **No Finding vs Cardiomegaly**. Each uses seed 43, pretrained ResNet-18, 20 epochs, batch size 32, Adam, and 224 × 224 images. The best epoch is selected by validation AUROC, and the selected checkpoint is evaluated on test once.

| Task | Best epoch | Validation AUROC | Test AUROC | Test examples |
|---|---:|---:|---:|---:|
| No Finding vs Pleural Effusion | 12 | 0.9497 | **0.9476** | 18,826 |
| No Finding vs Cardiomegaly | 19 | 0.9343 | **0.9377** | 8,205 |

These are single-seed results. They do not provide a between-seed variance estimate or reproduce the paper's top-five-of-ten summary. This run should be described as an **adapted reproduction**, not an exact replication. Details, epoch histories, comparisons, and limitations are in the [full experiment report](docs/resnet18-chexpert-experiment-report.md).

### Data source and cohort

Images were read from the public Hugging Face mirror [`danjacobellis/chexpert`](https://huggingface.co/datasets/danjacobellis/chexpert). The mirror's training records are stored as 23 Parquet shards with embedded JPEG bytes. The preparation script selects the project's existing CheXpert train/validation/test assignments; it does not construct a new random split. The required `chexpert.sample.train.csv`, `chexpert.sample.val.csv`, and `chexpert.sample.test.csv` files must be available under `datafiles/chexpert/` (or supplied to preparation using `CXR_CSV_SOURCE_DIR`). These split metadata files are not included as large dataset assets in this repository. Obtain approved project split artifacts and dataset access before running.

For each task, only rows where exactly one of `No Finding` and the target finding is positive are retained. The target finding is class 1 and No Finding is class 0. The three split files define the assignment; the Hugging Face mirror's small validation split is not used as this experiment's validation set. The selected 71,096 images were matched to the project metadata, with no path, disease-label, or split mismatches reported in the preparation audit.

### Preprocessing and model

Run `python chexpert/prepare_hf_baseline.py` from the repository root after obtaining the Parquet shards and split CSVs. The script reads the required image bytes, resizes grayscale images to 224 × 224, converts to `uint8`, and writes PNGs beneath `chexpert/hf_preprocessed/`. The loader repeats grayscale into three channels. Training uses random horizontal flipping and random affine transforms (rotation up to 15 degrees and scale 0.9–1.1); validation and test are unaugmented. This pipeline does not apply ImageNet mean/std normalization.

The model uses torchvision `ResNet18_Weights.IMAGENET1K_V1`, replaces the original final layer with an identity mapping, and adds a two-logit linear classifier. Adam trains the backbone at learning rate `0.001` and the classifier at `0.0001`. The repository runner uses cross-entropy with two logits, whereas the paper describes binary cross-entropy. On TPU it uses bfloat16 autocast and a BatchNorm running-statistics hook for the TorchTPU backend used in this run. These are material implementation choices; consult the report and runbook before interpreting or comparing the metrics.

### Obtain data and run

Download the Hugging Face training shards (this includes image payloads and requires substantial storage):

```bash
python -m pip install 'datasets>=3,<5' 'pyarrow>=18,<24'
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='danjacobellis/chexpert', repo_type='dataset', local_dir='chexpert/hf', allow_patterns=['data/train-*.parquet'])"
```

Review the dataset card and its terms. Ensure that the split CSVs are in `datafiles/chexpert/`, then prepare selected images:

```bash
python chexpert/prepare_hf_baseline.py
```

Activate the TPU-capable `crl-cxr` environment and launch the one-seed run from the repository root:

```bash
conda activate crl-cxr
env \
  PYTHONPATH="$PWD" \
  CXR_DEVICE=tpu \
  CXR_TPU_AMP=1 \
  CXR_NUM_WORKERS=4 \
  CXR_SEEDS=43 \
  CXR_TASKS='Pleural Effusion,Cardiomegaly' \
  python chexpert/run_resnet18_baseline.py
```

`CXR_SEEDS` and `CXR_TASKS` accept comma-separated values. The runner's default is seeds 42–51 and both tasks; explicitly narrow these values when reproducing the reported one-seed results. The run output is written to the git-ignored `chexpert/runs/resnet18_baseline_tpu_ce/`, including `config.json`, per-epoch `history.json`, selected checkpoints, validation/test prediction CSVs, and summaries. Do not commit images, patient-level CSVs, model checkpoints, credentials, or generated run artifacts.

For troubleshooting TPU access, dependency versions, data audit details, and the BatchNorm backend workaround, see the [runbook](docs/resnet18-chexpert-tpu-runbook.md). For the paper's broader methods and protocol deviations, see the [replication guide](docs/replication-guide.md) and [CheXpert Table 1 plan](docs/chexpert-table1-training-plan.md).

## Other experiment entry points

The project exposes the `race_invariance` and `view_invariance` Poetry commands, with implementation in `prediction/race_invariance.py` and `prediction/view_invariance.py`. The CheXpert Table 1 runner is `python scripts/run_chexpert_table1.py`; inspect its `--help` and the plan before starting a full matrix, as it launches many training jobs. Cluster shell scripts under `scripts/` may contain environment-specific assumptions and should be reviewed before use. `scripts/test.sh` is a manually configured training launcher, not an automated unit-test suite.

## Reproducibility and responsible data handling

Record the code revision, source dataset revision, split files, seed, software/device versions, command, and output hashes with new runs. The current baseline's dataset revision hash and some provenance hashes were not captured; the report lists these gaps and additional deviations. Follow the source dataset's access and use terms. Keep `.env`, credentials, private metadata, images, and generated artifacts out of Git.
