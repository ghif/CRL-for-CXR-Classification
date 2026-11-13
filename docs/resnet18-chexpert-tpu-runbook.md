# ResNet-18 CheXpert baseline on TorchTPU

This runbook records the steps used to prepare the Hugging Face CheXpert mirror and start the two-task ResNet-18 baseline from the project workspace. The current execution is an **adapted reproduction**: it uses the public Hugging Face image mirror and an explicitly validated TorchTPU BatchNorm workaround. It is not the paper's complete Table 1 matrix.

## 1. Scope and protocol

The run covers the ordinary classification baseline for two binary tasks:

- No Finding versus Pleural Effusion
- No Finding versus Cardiomegaly

The paper-inspired settings are 224 × 224 inputs, an ImageNet-pretrained ResNet-18, Adam, backbone learning rate `0.001`, classifier learning rate `0.0001`, batch size 32, and 20 epochs. The paper reports ten seeds (`42`–`51` here as a reproducibility choice) and selects the top five by validation AUROC for its summary. The user later narrowed this execution to one result per task to get an initial baseline sooner. The final run therefore uses seed 43 for each task and reports individual validation/test AUROCs; it does not estimate a multi-seed mean or standard deviation.

The paper specifies binary cross-entropy, while the repository's ResNet implementation uses a two-logit head and `CrossEntropyLoss`. The current runner follows the repository implementation; this is a protocol deviation to disclose with results. The paper also does not specify seed values or an unambiguous within-seed checkpoint epoch rule. Seeds 42–51 and max validation AUROC are recorded choices, not claims about paper-specified values. See [the broader replication plan](chexpert-table1-training-plan.md) for other paper ambiguities and the full Table 1 scope.

## 2. Workspace and environment

Run the commands from a clone of this repository. The experiment used a conda environment named `crl-cxr`:

```bash
conda activate crl-cxr
cd "$(git rev-parse --show-toplevel)"
```

TorchTPU setup is recorded in [`environment-torchtpu.yml`](../environment-torchtpu.yml). This environment uses CPU PyTorch wheels plus the `torch-tpu` plugin; the model is explicitly moved to the TPU at runtime. The installed versions during this run were Python 3.13, PyTorch `2.15.0+cpu`, torchvision `0.29.2+cpu`, `torch-tpu` `0.2.0`, and `libtpu` `0.0.52`. The Hugging Face Parquet reader dependencies `datasets` and `pyarrow` were also installed in this environment.

In the existing `crl-cxr` environment, the Parquet dependencies can be installed with:

```bash
python -m pip install 'datasets>=3,<5' 'pyarrow>=18,<24'
```

Check versions and device visibility before launching:

```bash
python -c 'import torch, torch_tpu, importlib.metadata as m; print("torch", torch.__version__); print("torchvision", m.version("torchvision")); print("torch-tpu", m.version("torch-tpu")); print("libtpu", m.version("libtpu")); print("accelerator", torch.accelerator.current_accelerator()); print("count", torch.accelerator.device_count()); print("available", torch.accelerator.is_available())'
```

In this environment the sandboxed probe reported no device, while the same probe outside the sandbox found one TPU. Run the TPU probe and training process in an execution context that can access the host TPU runtime. Do not silently let `CXR_DEVICE=tpu` fall back to CPU: the runner raises an error if TorchTPU is unavailable.

The runtime logged a PJRT API version difference (`0.117` in the plugin and `0.116` in the PyTorch framework). Device initialization and training nevertheless worked. Preserve this warning in experiment provenance if the environment is rebuilt or updated.

## 3. Obtain and inspect the Hugging Face mirror

The dataset repo used was [`danjacobellis/chexpert`](https://huggingface.co/datasets/danjacobellis/chexpert). Its training Parquet shards contain both row metadata and image bytes. The table reported 223,414 train rows across 23 shards and 234 validation rows. The local split comes from the project's existing CheXpert sample CSVs; the runner's train/validation/test examples are therefore selected from the mirror's train records, not the mirror's small validation split.

The train shards were downloaded under the ignored local `chexpert/` directory:

```bash
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='danjacobellis/chexpert', repo_type='dataset', local_dir='chexpert/hf', allow_patterns=['data/train-*.parquet'])"
```

This fetched the Parquet records, including image payloads, rather than just metadata. The HF card encodes labels as 0 = unlabeled, 1 = uncertain, 2 = absent, and 3 = present. Its AP/PA code interpretation is uploader-provided. Before use, the HF paths and labels were matched against the repository staging metadata and split CSVs: all 71,096 selected image paths matched, disease labels matched, and split assignments had zero mismatches. The AP/PA mapping also had zero mismatches. Sex had one mismatch per split and is not used in this baseline.

Review the dataset card and its terms before redistribution. The image files and row-level manifests remain local and ignored by git.

## 4. Prepare the selected images and CSVs

The preparation script is [`chexpert/prepare_hf_baseline.py`](../chexpert/prepare_hf_baseline.py). It reads the project's `chexpert.sample.{train,val,test}.csv` files from the existing local staging tree, maps their `path_preproc` values to HF paths, scans Parquet path columns, and extracts only the 71,096 paths needed by the two tasks and three splits.

The script resizes the grayscale JPEG bytes to 224 × 224 with `skimage.transform.resize(..., preserve_range=True)`, converts to `uint8`, and writes PNGs under `chexpert/hf_preprocessed/preproc_224x224/`. It copies the three sample CSVs into `datafiles/chexpert/` for the runner. Launch it from the repository root:

```bash
python chexpert/prepare_hf_baseline.py
```

The completed `chexpert/preparation.json` reported:

- 223,414 rows in the HF train split
- 71,096 selected paths, all extracted and present
- 32 comparison images, zero exact pixel matches against the previously staged copies
- mean SSIM 0.9467 and minimum SSIM 0.9247 for those 32 comparisons

Thus the evaluation uses the HF image bytes, not the differing previously staged image files. The loader repeats each grayscale image into three channels for ResNet. The repository `CXRDataset` supplies its training augmentation (random horizontal flip and random affine rotation/scale) and reads image intensities as float values from the saved PNG; this run does not add ImageNet mean/std normalization.

The local paths under `chexpert/` and `datafiles/chexpert/` are ignored by git. Keep them, checkpoints, and prediction files out of commits and public artifacts.

For each task, `CXRDataset` retains records where exactly one of `No Finding` or the task label equals 1. Rows where neither label is 1 and rows where both are 1 are excluded. The target is 1 for the disease and 0 for No Finding. It maps AP to view code 0 and PA to view code 1, and the baseline allows both codes (`protected_race_set=[0, 1]`); no race or sex subgroup objective is used.

## 5. TPU preflight and BatchNorm behavior

Before the full run, a small ResNet forward/backward pass on `tpu` succeeded with bfloat16 autocast and cross-entropy. A direct BatchNorm probe then showed a backend problem in the installed TorchTPU build: a training `BatchNorm2d` call on constant input left `running_mean` at 0 and `running_var` at 1. Without valid running statistics, evaluation produced nearly saturated scores and AUROC around 0.50, despite falling training loss.

The runner therefore installs a forward-pre-hook on each of ResNet-18's 20 BatchNorm layers. For each training batch, the hook computes per-channel mean and unbiased variance over batch and spatial dimensions, then applies the normal PyTorch exponential moving average using the layer's momentum. A TPU probe confirmed the expected update (mean 0 → 0.5 and variance 1 → 0.9 for constant input 5 with momentum 0.1). With this hook, validation AUROC on Pleural Effusion seed 42 was 0.937–0.948 instead of about 0.50. This is an implementation workaround for the installed TPU backend, and is disclosed in each run's `config.json`.

For a minimal device/backprop check, use the same model construction as `make_model()` in the runner, move the backbone and head to `torch.device("tpu")`, and run one batch under:

```python
with torch.autocast("tpu", dtype=torch.bfloat16):
    logits = classifier(backbone(images))
    loss = torch.nn.functional.cross_entropy(logits.float(), labels.long())
loss.backward()
```

The TPU hook and AUROC behavior should be checked again after changing TorchTPU, PyTorch, or libtpu versions. TorchTPU's AMP interface is documented in its [API reference](https://torchtpu.readthedocs.io/en/latest/api_reference/); the BatchNorm running-stat issue is tracked in the [TorchTPU issue tracker](https://github.com/google-pytorch/torch_tpu/issues/6465).

## 6. Launch the experiment

From the repository root, launch the one-seed-per-task run in the TPU-accessible host context:

```bash
env \
  PYTHONPATH="$PWD" \
  CXR_DEVICE=tpu \
  CXR_TPU_AMP=1 \
  CXR_NUM_WORKERS=4 \
  CXR_SEEDS=43 \
  CXR_TASKS='Pleural Effusion,Cardiomegaly' \
  python chexpert/run_resnet18_baseline.py
```

`PYTHONPATH` is required because the script imports `prediction.datasets.cxr` while it is located under `chexpert/`. `CXR_DEVICE=tpu` selects the accelerator, `CXR_TPU_AMP=1` enables TPU bfloat16 autocast for model forward passes, four loader workers feed the TPU, and `CXR_SEEDS`/`CXR_TASKS` narrow the run. Remove those final two variables to run the original ten-seed schedule for both tasks. Inputs and labels are transferred to the TPU; predictions and checkpoints are copied back to CPU for serialization and AUROC computation. In the one-seed run, test evaluation occurs after the single validation-selected checkpoint is ready.

The runner prints one JSON record per task, seed, and epoch. It uses train, validation, and test counts after the project's binary cohort filters. The observed Pleural Effusion counts are 37,529 train, 6,135 validation, and 18,826 test examples; training drops the last incomplete batch, so 37,504 examples contribute to each epoch's loss.

## 7. Output layout and interpretation

The active output root is:

```text
chexpert/runs/resnet18_baseline_tpu_ce/
```

Expected outputs:

```text
config.json
pleural_effusion/seed_42/best.pt
pleural_effusion/seed_42/history.json
pleural_effusion/seed_42/predictions.val_best.csv
...
pleural_effusion/summary.json
cardiomegaly/...
summary.json
```

Each completed seed stores its best validation-AUROC checkpoint, per-epoch training/validation history, and continuous positive-class validation scores. Each evaluated seed also gets `predictions.test.csv`. Per-task summaries report selected seeds and test AUROCs. A sample standard deviation is undefined for one seed and is stored as `null`. `config.json` records hyperparameters, source, accelerator, AMP, BatchNorm hook count, and deviations.

Do not treat the earlier interrupted `resnet18_baseline/` artifacts as final results: that first attempt used one-logit BCE, ran on CPU, and was stopped. An intermediate TPU attempt used the correct two-logit head but had stale BatchNorm statistics and was also stopped. The current `resnet18_baseline_tpu_ce/` run is the active one with TPU, cross-entropy, and the BatchNorm workaround.

## 8. Completed one-seed results recorded on 2026-11-13

The initial ten-seed job was stopped during Pleural Effusion seed 44 at epoch 12, at the user's request to stop the multi-seed schedule. That incomplete seed produced no completed checkpoint and is not included. Completed Pleural Effusion seed 43 was reused; it had the highest validation AUROC among the two completed seeds (42 and 43). Cardiomegaly seed 43 was trained for 20 epochs. Each task was evaluated on its test split once:

| Task | Seed | Best epoch | Validation AUROC | Test AUROC | Test examples |
|---|---:|---:|---:|---:|---:|
| No Finding vs Pleural Effusion | 43 | 12 | 0.9497 | 0.9476 | 18,826 |
| No Finding vs Cardiomegaly | 43 | 19 | 0.9343 | 0.9377 | 8,205 |

These are single-seed baseline results, not the paper's top-five-of-ten mean ± standard deviation. Pleural Effusion seed 43 was selected as best among only the two completed candidates before the run was narrowed, not among all ten seeds. Cardiomegaly has one seed only. The test split was not used to select either checkpoint. The output root's `summary.json` and each seed's prediction/history/checkpoint files contain the machine-readable results. Retain the data-source, protocol, and TorchTPU BatchNorm deviations stated above when reporting them.
