# Replicating the CXR experiments

This guide describes a careful reproduction of the chest-radiograph experiments in this repository and explains how to map them to the original paper. It is based on the checked-in implementation and experiment scripts. **The paper PDF, supplement, and a full paper citation are not included in this checkout** (the top-level README has no citation), so do not treat repository defaults or script comments as proof of paper settings. Before claiming an exact paper reproduction, obtain the canonical paper and supplement, record their bibliographic details, and fill the paper-verification checklist below from those sources. Where the implementation and paper differ, report both rather than silently choosing one.

## 1. What is being reproduced

The repository implements binary CXR classification experiments on **CheXpert** and **MIMIC-CXR**, with an optional cross-dataset evaluation. It supports:

- target disease: `Cardiomegaly` or `Pleural Effusion`;
- protected attribute: race, sex, or radiographic view;
- ImageNet-pretrained DenseNet-121, ResNet-18/34/50, ViT-B/16, and EfficientNet-B0;
- ordinary classification and an invariant-sampling training path;
- in-domain test evaluation and a separate-dataset (`ood`) evaluation;
- subgroup metrics, predictions, logits, and embeddings.

The repository's invariant objective is implemented in `prediction/backbones.py`: for a list of examples treated as a group, it sums disease-classification losses, computes pairwise-in-sequence embedding MSE between adjacent examples, subtracts the mean embedding entropy, and multiplies the invariant term by `inv_loss_coefficient`. Thus the code's implemented objective is approximately

`sum_i L_class(i) + lambda * (sum_adjacent MSE(z_i, z_(i-1)) - sum_i H(softmax(z_i)))`.

This precise implementation should be cited when describing what was run; do not substitute a generic contrastive or domain-adversarial loss. `prediction/datasets/cxr.py` constructs the groups by selecting a common disease label and sampling images from the chosen protected-attribute categories. See the implementation notes and audit checklist below: the source has consequential details that need checking against the paper before interpreting this as a faithful implementation of the paper's method.

## 2. Freeze and document the experimental target

Before running jobs, create a run sheet (or machine-readable config) for every experiment. Record:

| Item | Record explicitly |
|---|---|
| Paper | Full title, authors, venue/year, DOI/arXiv URL, version, supplement, and the table/figure being reproduced |
| Task | Dataset, disease, protected attribute, source/target domain, and exact train/test subgroup definitions |
| Data | Dataset release, access date, official split files, image preprocessing/version, label policy, exclusions, and counts per split and subgroup |
| Method | Ordinary vs invariant-sampling mode, group size `nsamples`, invariant-loss coefficient, backbone, pretrained-weight version, and any paper-specific changes |
| Optimization | Seed, optimizer, learning rates, batch size, epochs/steps, checkpoint criterion, image size, augmentation, hardware, and software versions |
| Evaluation | Checkpoint selection, threshold, exact metric definitions, subgroup aggregation, confidence intervals, and evaluation split |
| Provenance | Git commit, command/config, environment lockfile, logs, checkpoint hash, and generated-output hashes |

Treat the paper and its supplement as the source of truth for values not stated in the code. Treat the code as the source of truth for the actual behavior of this repository. The CLI defaults (`seed=42`, `size=224`, `batch_size=32`, `epochs=10`, `nsamples=1`, and coefficient `1`) are conveniences, **not evidence that those are the paper's settings**.

### Paper-verification checklist (complete before making an exact-reproduction claim)

- [ ] Confirm the intended paper/version and identify its CXR experiments, tables, and supplementary protocols.
- [ ] Transcribe the paper's patient-level split rules, dataset releases, label treatment, demographic mappings, and image preprocessing.
- [ ] Transcribe each reported backbone, initialization, optimizer, learning rates, schedule, epochs, batch/group size, augmentation, and seed count.
- [ ] Verify the paper's mathematical objective and sampling procedure against `BaseNet.process_batch_list` and `CXRDataset.get_samples`.
- [ ] Verify model-selection and metric definitions, including thresholds and subgroup averaging.
- [ ] Compare the paper protocol with the actual commands/configs and document every deviation.

## 3. Environment and data

### Software

The project pins Python `3.12.0`, PyTorch `2.4.1`, torchvision `0.19.1`, Lightning `2.4.0`, NumPy `1.26.3`, scikit-learn `1.6.1`, and the remaining direct dependencies in `pyproject.toml`/`poetry.lock`. Use the lockfile and record the actual Python, CUDA, driver, GPU, and package versions. Install GPU dependencies only as appropriate for the host; do not assume the optional CUDA package group is a complete substitute for a compatible PyTorch/CUDA installation.

```bash
poetry install
poetry run race_invariance --help
```

For a Google TPU VM using TorchTPU, use the separate [`environment-torchtpu.yml`](../environment-torchtpu.yml) environment recipe instead of Poetry's pinned stack:

```bash
conda env create --solver classic -f environment-torchtpu.yml
eval "$(conda shell.bash hook)"
conda activate crl-cxr-torchtpu
```

The TorchTPU recipe intentionally uses Python 3.13, CPU PyTorch 2.15, torchvision 0.29, NumPy 2, and TensorBoard 2.21; it is not a recreation of `poetry.lock`. The CXR race-invariance entry point auto-selects an available TorchTPU device or accepts `--accelerator tpu`. This implementation targets one TPU device per process on this VM; it does not configure multi-host or multi-device distributed training. To fail rather than fall back to CPU, pass `--accelerator tpu` explicitly. From the repository root, launch it with the module entry point:

```bash
python -m prediction.race_invariance \
  --accelerator tpu --dataset-train chexpert --dataset-test mimic \
  --disease 'Pleural Effusion' --invariance race \
  --model-type densenet --seed 42 --epochs 10 --batch-size 32
```

The repository's Slurm examples assume the authors' cluster paths, partitions, usernames, conda environment, and environment variables. They are examples, not portable submission scripts. Adapt them to the local scheduler and verify the expanded command before submitting.

### Data access and paths

CheXpert and MIMIC-CXR are separately governed datasets. Obtain them through their official access processes, comply with their terms, and do not commit images, patient-level tables, credentials, or private paths. The project reads `CHEXPERT_FOLDER`, `MIMIC_FOLDER`, and `MIMIC_CXR_FOLDER` (and `MIMIC_IV_FOLDER` for the setup helper) from `.env`. Keep this file private. `datafiles/setup_data.py` copies selected metadata tables into `datafiles/`; it does **not** download or preprocess the image collections.

The training code expects split CSVs at:

```text
datafiles/<dataset>/<dataset>.sample.train.csv
datafiles/<dataset>/<dataset>.sample.val.csv
datafiles/<dataset>/<dataset>.sample.test.csv
```

It joins each row's `path_preproc` to the configured image root. The only checked-in CXR CSV is `datafiles/chexpert/chexpert_split_2021_08_20.csv`; the large sample split CSVs, MIMIC splits, and image preprocessing outputs are not present in the listed repository files. Locate the corresponding approved study data and project-provided artifacts (the README links a Box folder), then verify all paths and file counts before a full run. Never assume the tiny split preview is a complete dataset or reconstruct the study split from a random split without explicit evidence from the paper.

### Data integrity gate

Before training, verify and save a data audit containing:

1. split CSV existence, row count, unique patient/study/image identifiers, duplicate and cross-split leakage checks, and hashes;
2. existence/readability of every referenced image and the number of missing files;
3. label values, missingness policy, and the exact target inclusion counts;
4. protected-attribute values and mapping for each dataset, with counts by target and split;
5. image dimensions, dtype, intensity range, channel count, and the preprocessing that generated `path_preproc`;
6. explicit evidence that preprocessing and split construction match the paper.

The CXR loader keeps rows where exactly one of `No Finding` and the selected disease is positive. It then assigns the disease as a binary target. Other uncertain or co-occurring labels are not resolved by an explicit policy in this loader; inspect upstream CSV generation and compare with the paper. Do not infer race-code meanings from integer values. The data loader expects numeric protected labels and filters them against the supplied sets. Sex maps `Male` to 0 and `Female` to 1; view maps `AP` to 0 and `PA` to 1. Confirm source values, missing-value handling, and encoding before using these mappings. Unknown strings can raise errors.

## 4. Establish a faithful baseline first

Use the ordinary, non-invariant run as a control. Make the domain, disease, attribute, backbone, seed, and training settings identical to the corresponding invariant run; only change the invariant-sampling switch and its method-specific group/coefficient settings. Run a smoke test first, then full training.

Example CLI shape (replace values only after checking the paper and data audit):

```bash
poetry run race_invariance \
  --dataset-train chexpert --dataset-test mimic \
  --disease 'Pleural Effusion' --invariance race \
  --model-type densenet --seed 42 --size 224 \
  --epochs 10 --batch-size 32 --gpus 1 --dev 0 \
  --protected-race-set-train 0 1 2 \
  --protected-race-set-test 0 1 2
```

This is a syntax example, not a paper-verified recipe. Check `race_invariance --help` for supported options. For same-dataset experiments, use the same dataset for train and test; for cross-dataset experiments, use the intended source as `--dataset-train` and target as `--dataset-test`. `dataset-test` supplies the `ood` evaluation loader; the routine also reports source validation and source test. Do not select hyperparameters using target-domain `ood` results.

Run a bounded smoke test before a costly run:

```bash
poetry run race_invariance \
  --dataset-train chexpert --dataset-test chexpert \
  --disease 'Pleural Effusion' --invariance race \
  --model-type densenet --test --gpus 1
```

`--test` is an engineering smoke mode, not a valid abbreviated estimate of final performance: it limits training to five steps, validation cadence, and evaluation to only a few batches. Confirm that data load, forward/backward, checkpoint creation, and output writing work. Then remove `--test` for actual reproduction.

## 5. Run the invariant experiment

The repository enables the grouped path with `--invariant-sampling`. `nsamples` controls the number of examples returned in each group. `--inv-loss-coefficient` controls the weight of the implemented invariant term. For example:

```bash
poetry run race_invariance \
  --dataset-train chexpert --dataset-test mimic \
  --disease 'Pleural Effusion' --invariance race \
  --model-type densenet --seed 42 --size 224 \
  --epochs 10 --batch-size 32 --gpus 1 --dev 0 \
  --protected-race-set-train 0 1 2 \
  --protected-race-set-test 0 1 2 \
  --invariant-sampling --nsamples 2 --inv-loss-coefficient 1
```

Again, parameter values above illustrate the interface; use the paper's values when verified. The experiment scripts contain commented example sweeps, not a definitive complete reproduction matrix. `scripts/race_invariance_more.sh` currently has active commands near its end for sex experiments and seeds 44–49; do not execute it blindly if the intended study is different. Inspect scripts before running; many examples are commented and some refer to unsupported model names or legacy argument shapes.

### Important implementation checks

These checks are especially important because they affect what the code actually trains:

- `CXRDataset.get_samples` chooses one disease class and samples each group member's protected category using `np.random.choice(self.protected_race_set, self.nsamples)`. It does not use the computed `protected_race_probs` when choosing the categories. It assumes every selected protected category contains at least one example of the selected disease class; empty category/class cells can fail at sampling time.
- `get_samples` calls `np.random.seed(item)`, making sample selection deterministic by dataset index and resetting NumPy's global RNG each access. Check the implications for worker behavior and augmentation randomness; document it rather than assuming independent stochastic resampling each epoch.
- `process_batch_list` orders group members, sums their class losses, and only applies embedding MSE to consecutive pairs. It does not form a full all-pairs loss. It subtracts entropy of `softmax(embedding)` (embedding coordinates interpreted as logits), which is a specific implementation detail requiring mathematical comparison with the paper.
- The grouped loader applies augmentation independently per image, but the `RandomAffine`/flip input shape and interpolation behavior should be validated with a real batch. `image_size` is passed into the dataset but this loader does not visibly resize images; confirm that `path_preproc` images are already the intended size.
- `batch_size` counts groups in invariant mode; the number of images processed per batch is approximately `batch_size * nsamples`. The training loader drops the final incomplete batch and sets `prefetch_factor=5`.
- The constructor's defaults and CLI options are not wholly aligned (for example, the CLI protected-set defaults include category 3, while the `CXRDataset` race default is `[0, 1, 2]`). Pass the intended set explicitly and verify actual subgroup counts.
- Validation/test use the ordinary one-image path, even when training uses grouped sampling. This is appropriate only if it matches the intended evaluation protocol.

For a faithful reproduction, resolve any discrepancy through evidence from the paper, published artifacts, or a checked-in author protocol. If a code correction is required, preserve a reference run of the unmodified implementation and report the patch and its effect separately.

## 6. Checkpoints, metrics, and outputs

The code uses Lightning's `ModelCheckpoint` monitoring `val_loss_class` in `min` mode and evaluates the best checkpoint for ordinary runs (the last checkpoint in `--test` mode). The CXR trainer checks validation every 300 training steps and saves checkpoints every 301 training steps; this is not necessarily equivalent to an epoch-based schedule. Record checkpoint path/hash and the validation event used to select it.

Runs are written below `logs/` (or `logs_test/`) with paths encoding task, model, seed, dataset transfer, and invariant settings. Each evaluation split writes:

- `count_info_<split>.json`: global and protected-subgroup counts;
- `predictions.<split>.csv`: predicted class, logits, target, and protected value;
- `embeddings.<split>.csv`: embedding coordinates, target, and protected value;
- `metrics.<split>.csv`: metrics emitted by `prediction/metrics.py`.

The available splits are `val`, source `test`, and target `ood`. The metrics implementation currently constructs ROC AUC from the **hard predicted class**, not the positive-class probability/logit, and returns F1, accuracy, recall, precision, ROC AUC, TPR, and FPR. This differs from the usual threshold-independent ROC AUC calculation and must be reconciled with the paper before comparing published AUC. It also uses class predictions from argmax (equivalent to a fixed 0.5 boundary for a two-logit softmax only under the usual comparison), and subgroup metric robustness depends on both classes being present. Validate metric calculations independently against saved logits and paper definitions; report any corrected metrics separately from the repository's raw output.

Do not report a single aggregate alone when the paper reports subgroup outcomes. Include sample counts, metric uncertainty (using the paper's method if specified), and the exact aggregation across categories/seeds. Keep validation-based model selection separate from final test reporting. Never tune on `test` or `ood`.

## 7. Repeated runs and comparisons

For every paper result, reproduce the complete set of required datasets, attributes, diseases, backbones, transfer directions, and seeds stated by the paper. Use a controlled comparison table with one row per run and include at least:

- paper table/figure and experiment identifier;
- ordinary vs invariant method;
- source/target data and protected groups;
- seed, model, all CLI values, and code commit;
- data counts by split and subgroup;
- selected checkpoint and validation criterion;
- global and subgroup metrics, uncertainty, and output artifact locations;
- deviations, runtime environment, failures, and exclusions.

Avoid comparing runs with different splits, preprocessing, initialization, training budget, or evaluation threshold as if only invariance changed. If reproducing a cross-dataset result, explicitly distinguish source-domain validation/test from target-domain evaluation. Publish enough configuration and provenance to rerun the experiment without publishing restricted patient-level data.

## 8. Troubleshooting and reproducibility hygiene

- **Missing file / path errors:** verify `.env`, split CSV columns, and the joined `img_data_dir + path_preproc` path. The code concatenates strings, so ensure the root includes the intended separator.
- **Empty subgroup sampling:** print counts for every target-by-protected-category cell before invariant training; remove or revise categories only if the protocol permits, and record the change.
- **Different image shapes:** inspect preprocessing outputs and confirm fixed dimensions/channels; the CXR loader itself does not resize or normalize images in the shown code.
- **OOM:** remember the grouped batch processes `batch_size * nsamples` images; adjust only with a documented protocol change.
- **Unstable/irreproducible runs:** record all seeds and hardware; verify Lightning worker seeding, NumPy reseeding in `get_samples`, nondeterministic GPU kernels, and data-loader worker settings.
- **Metric exceptions or NaNs:** verify both classes occur in each evaluated subgroup and check the metric implementation's ROC-AUC assumptions.
- **Checkpoint missing:** confirm validation is running, `val_loss_class` is logged, and the run reaches a checkpoint save event. Preserve logs and configuration on failure.

Keep per-run logs, configs, output CSVs, and checkpoints together. Hash source revision, split files, and final artifacts. Do not overwrite a successful run when testing a code or preprocessing change.

## 9. Reporting the result honestly

Use one of these labels:

1. **Exact protocol reproduction** only when the paper and supplement settings were verified and the data, code behavior, and reported metrics match them (with all deviations resolved).
2. **Repository implementation reproduction** when the checked-in code was run as written but exact alignment with the paper is unverified.
3. **Adapted reproduction** when data, code, training settings, metric definitions, or compute constraints differ; enumerate every material adaptation.

The present checkout alone supports planning a repository-implementation reproduction, but it does not contain enough paper-specific information to certify an exact paper reproduction. The linked model/output archive in the README may help cross-check artifacts, but it is not a replacement for the paper's protocol or data provenance.

## References in this checkout

- Project source: `prediction/race_invariance.py`, `prediction/backbones.py`, `prediction/datasets/cxr.py`, `prediction/metrics.py`, `datafiles/setup_data.py`.
- Project configuration: `pyproject.toml`, `poetry.lock`.
- Cluster/example launch scripts: `scripts/train_job_race_invariance.sh`, `scripts/race_invariance_more.sh`.
- Dataset/artifact pointer: repository `README.md` (Box-hosted models, embeddings, and predictions).
- **Original paper citation:** not present in this checkout; add the verified canonical citation and URL here before distributing this guide as a paper-specific protocol.
