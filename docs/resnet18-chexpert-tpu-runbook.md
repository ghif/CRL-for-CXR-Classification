# ResNet-18 CheXpert baseline on TorchTPU

This is the manual, end-to-end procedure for reproducing the repository's two-task ResNet-18 baseline from the Hugging Face CheXpert mirror. It covers the exact single-seed run recorded in this repository and the checks needed before launching it. The result is an **adapted reproduction**, not the paper's full Table 1 replication; protocol differences are listed below and in the [experiment report](resnet18-chexpert-experiment-report.md).

## 1. Experiment being reproduced

The two independent binary tasks are No Finding versus Pleural Effusion and No Finding versus Cardiomegaly. The reported run uses seed 43 for each task, an ImageNet `IMAGENET1K_V1` ResNet-18 backbone, 224 × 224 inputs, batch size 32, Adam for 20 epochs, backbone learning rate `0.001`, and classifier learning rate `0.0001`. Each task's checkpoint is chosen by maximum validation AUROC; test AUROC is calculated once from that checkpoint.

The paper reports ten seeds and a top-five summary. The run documented here has one seed per task. It also uses the Hugging Face mirror, a two-logit cross-entropy objective (the paper describes binary cross-entropy), and a TorchTPU BatchNorm statistics workaround. Do not interpret its scores as a like-for-like comparison with the paper.

| Task | Train | Validation | Test | Best epoch | Test AUROC |
|---|---:|---:|---:|---:|---:|
| No Finding vs Pleural Effusion | 37,529 | 6,135 | 18,826 | 12 | 0.9476 |
| No Finding vs Cardiomegaly | 15,867 | 2,634 | 8,205 | 19 | 0.9377 |

Counts above are after the loader's cohort filtering. The training loader drops the last incomplete batch.

## 2. Requirements and workspace setup

Use a Linux host with a TPU runtime visible to TorchTPU, sufficient local storage for the Parquet shards and 71,096 extracted PNGs, and this repository checked out. The run was recorded with Python 3.13, PyTorch `2.15.0+cpu`, torchvision `0.29.2+cpu`, `torch-tpu` `0.2.0`, and `libtpu` `0.0.52`. The project Poetry environment is separate and is not the environment used for this run.

From a shell with the host's conda installation available, activate the existing TPU environment or create one from the repository recipe. For the existing environment used in the recorded run:

```bash
source /mnt/data/miniconda/etc/profile.d/conda.sh
conda activate crl-cxr
cd /mnt/data/firstmate/projects/CRL-for-CXR-Classification
```

If setting up a new environment instead, the checked-in recipe names it `crl-cxr-torchtpu`:

```bash
conda env create --solver classic -f environment-torchtpu.yml
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate crl-cxr-torchtpu
python -m pip install 'huggingface_hub>=0.30' 'datasets>=3,<5' 'pyarrow>=18,<24'
```

The first command installs TorchTPU and the core project libraries. The final command installs the Hugging Face downloader and Parquet reader used for data acquisition and extraction. Record actual versions when reproducing; package updates can change behavior.

Check the interpreter, packages, and TPU before downloading data:

```bash
which python
python --version
python -c 'import torch, torch_tpu, torchvision, importlib.metadata as m; print("torch", torch.__version__); print("torchvision", torchvision.__version__); print("torch-tpu", m.version("torch-tpu")); print("libtpu", m.version("libtpu")); print("accelerator", torch.accelerator.current_accelerator()); print("count", torch.accelerator.device_count()); print("available", torch.accelerator.is_available())'
```

Proceed only when the current accelerator is TPU, the device count is nonzero, and `available` is true. The runner deliberately raises an error if `CXR_DEVICE=tpu` is requested without an available TPU. A CPU-only shell/container may not expose the host TPU even when the machine has one; run the commands in the TPU-enabled host context. The recorded host logged a PJRT API version difference (plugin 0.117, framework 0.116), although initialization and training worked.

## 3. Obtain the data and the project split files

The image source is the [`danjacobellis/chexpert` Hugging Face dataset](https://huggingface.co/datasets/danjacobellis/chexpert). Review its dataset card, access conditions, and redistribution terms before downloading or sharing data. The command below downloads the 23 training Parquet shards, which contain embedded image bytes; it is not a metadata-only download and needs substantial disk space.

```bash
mkdir -p chexpert
python -c "from huggingface_hub import snapshot_download; print(snapshot_download(repo_id='danjacobellis/chexpert', repo_type='dataset', local_dir='chexpert/hf', allow_patterns=['data/train-*.parquet']))"
```

Confirm the expected shards are present:

```bash
find chexpert/hf/data -maxdepth 1 -name 'train-*.parquet' -type f | wc -l
```

The expected count is 23. The downloaded train split has 223,414 rows. The mirror's separate 234-row validation split is **not** used for this experiment's validation set.

Obtain the existing project split CSVs from the approved project data/artifact source; the CSVs are ignored by Git and are not committed in this repository. Place these files in `datafiles/chexpert/`:

```text
chexpert.sample.train.csv
chexpert.sample.val.csv
chexpert.sample.test.csv
```

If they are stored elsewhere, set `CXR_CSV_SOURCE_DIR` to their directory before running the preparation script. Do not substitute random splits: preserving these assignments is necessary to reproduce the reported evaluation. Check the files before continuing:

```bash
for split in train val test; do
  test -s "datafiles/chexpert/chexpert.sample.${split}.csv" || exit 1
done
python -c 'import pandas as pd; from pathlib import Path; p=Path("datafiles/chexpert"); [(lambda d,s: print(s, "rows", len(d), "columns", len(d.columns), "path_preproc", "path_preproc" in d))(pd.read_csv(p/f"chexpert.sample.{s}.csv"), s) for s in ("train","val","test")]'
```

Each CSV must contain the `path_preproc` field and the target, `No Finding`, and loader-required metadata columns used by `prediction/datasets/cxr.py`. If preparation reports an unexpected path shape or count, stop and resolve the split/source mismatch; do not edit paths to force the script through.

## 4. Extract and preprocess the selected images

Run the preparation script from the repository root:

```bash
python chexpert/prepare_hf_baseline.py
```

The script reads the three split CSVs, converts their `path_preproc` values to Hugging Face paths, scans path columns in the Parquet shards, and extracts the 71,096 required images. Each grayscale JPEG is resized to 224 × 224 using `skimage.transform.resize(..., preserve_range=True)`, cast to `uint8`, and saved as PNG at the same relative path beneath `chexpert/hf_preprocessed/`. The CSV files are copied into `datafiles/chexpert/` if a separate `CXR_CSV_SOURCE_DIR` was supplied.

The script prints a JSON audit and writes `chexpert/preparation.json`. Confirm that `selected_paths` and `extracted_missing` are 71,096 and `selected_images_complete` is `true`. The recorded preparation found all selected paths and produced all images. A comparison of 32 images with an existing staged copy found no pixel-identical files (mean SSIM 0.9467; minimum 0.9247); the run consistently used the Hugging Face bytes.

The `CXRDataset` loader repeats grayscale values across three channels and reads them as float values without ImageNet mean/std normalization. It applies random horizontal flip and, with probability 0.5, an affine transform with rotation up to 15 degrees and scale 0.9–1.1 to training samples; validation and test samples are unaugmented. For each task, the loader keeps examples where exactly one of `No Finding` and the target disease is positive, assigns disease=1 and No Finding=0, and excludes rows where both/neither are positive. It does not use race or sex for this baseline.

## 5. Verify TPU execution and BatchNorm handling

The runner applies bfloat16 autocast to model forward passes when `CXR_TPU_AMP=1`; logits and loss are converted/calculated in float32. Before a long run, ensure the TPU check in section 2 succeeds. The recorded environment also passed a ResNet forward/backward smoke check on TPU.

The installed TorchTPU build did not update BatchNorm `running_mean` and `running_var` buffers during training. Stale statistics caused validation AUROC near 0.50. `chexpert/run_resnet18_baseline.py` installs a forward-pre-hook on the 20 ResNet-18 BatchNorm layers to compute channel mean and unbiased variance for each training batch and apply the PyTorch momentum update. A constant-input TPU probe verified the update. This workaround is part of the experiment; recheck it if changing TorchTPU, PyTorch, or libtpu versions. The issue is tracked in the [TorchTPU issue tracker](https://github.com/google-pytorch/torch_tpu/issues/6465).

## 6. Run one seed per task

Set the task and seed explicitly. The runner's defaults are seeds 42–51, so omitting `CXR_SEEDS=43` starts multiple seeds. Use the following command for the completed one-seed-per-task configuration:

```bash
set -o pipefail
PYTHONPATH="$PWD" \
  CXR_DEVICE=tpu \
  CXR_TPU_AMP=1 \
  CXR_NUM_WORKERS=4 \
  CXR_SEEDS=43 \
  CXR_TASKS='Pleural Effusion,Cardiomegaly' \
  python chexpert/run_resnet18_baseline.py 2>&1 | tee chexpert/resnet18-seed43.log
```

This runs the tasks sequentially on one TPU, each for 20 epochs. It writes one JSON progress record per epoch and prints post-filter split counts. The first run downloads ImageNet pretrained weights if they are not already cached. Keep the terminal open until the process exits with status 0; `set -o pipefail` ensures a failed Python process is not hidden by `tee`.

For only one task, set `CXR_TASKS='Pleural Effusion'` or `CXR_TASKS='Cardiomegaly'`. For a deliberately different seed, set `CXR_SEEDS` to that integer. To run the broader ten-seed protocol, specify `CXR_SEEDS='42,43,44,45,46,47,48,49,50,51'` knowingly: it is substantially longer and still does not make this an exact paper reproduction.

The runner reuses a completed seed when its checkpoint, history, and validation predictions exist. If an output folder already contains that seed, the console logs `reusing completed checkpoint`; the seed will not be retrained. The runner's output path is fixed at `chexpert/runs/resnet18_baseline_tpu_ce/`. Before intentionally repeating a seed from scratch, archive the existing output directory outside that path so old artifacts are not confused with new ones. Do not remove or overwrite outputs without first preserving any results you need.

## 7. Inspect outputs and verify the result

The expected output tree is:

```text
chexpert/runs/resnet18_baseline_tpu_ce/
├── config.json
├── summary.json
├── pleural_effusion/
│   ├── summary.json
│   └── seed_43/
│       ├── best.pt
│       ├── history.json
│       ├── predictions.val_best.csv
│       └── predictions.test.csv
└── cardiomegaly/
    ├── summary.json
    └── seed_43/...
```

Inspect the configuration and summaries:

```bash
python -m json.tool chexpert/runs/resnet18_baseline_tpu_ce/config.json
python -m json.tool chexpert/runs/resnet18_baseline_tpu_ce/pleural_effusion/summary.json
python -m json.tool chexpert/runs/resnet18_baseline_tpu_ce/cardiomegaly/summary.json
```

For the recorded seed-43 run, expect best epochs 12 and 19, validation AUROCs 0.9497 and 0.9343, and test AUROCs 0.9476 and 0.9377 for Pleural Effusion and Cardiomegaly respectively. Minor numerical differences can occur across software/runtime versions. Confirm `config.json` shows device `tpu`, AMP enabled, and 20 BatchNorm hooks. Check histories contain 20 epochs and that test prediction CSVs have the expected number of examples before quoting the metrics. The report contains the complete epoch histories and paper comparison.

## 8. Reproducibility notes and limitations

The source mirror's revision/hash, exact data file hashes, TPU model identifier, and a complete captured stdout log were not recorded for the original run. For new runs, save those along with Git revision, command, environment versions, split-file hashes, preparation JSON, and output hashes. Keep downloaded Parquet files, PNG images, patient-level CSVs, checkpoints, and logs local; they are under ignored paths and should not be committed or redistributed without permission.

The reported single-seed AUROCs (0.9476 Pleural Effusion; 0.9377 Cardiomegaly) are individual test results. The paper reports a top-five-of-ten summary. Other deviations include mirror image bytes, the repository's two-logit cross-entropy head instead of the paper's stated BCE, and the TPU BatchNorm workaround. See the [full report](resnet18-chexpert-experiment-report.md), [replication guide](replication-guide.md), and [Table 1 plan](chexpert-table1-training-plan.md) when comparing protocols.
