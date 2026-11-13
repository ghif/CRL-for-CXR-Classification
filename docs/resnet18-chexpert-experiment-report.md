# ResNet-18 Baseline Evaluation on Hugging Face CheXpert

## Abstract

We trained an ImageNet-initialized ResNet-18 baseline for two binary chest X-ray classification tasks using image records from the public Hugging Face mirror `danjacobellis/chexpert`: No Finding versus Pleural Effusion and No Finding versus Cardiomegaly. The source image records were matched to the project's existing train, validation, and test split CSVs, resized to 224 × 224 grayscale PNGs, and loaded as three-channel tensors. Training used Adam for 20 epochs with batch size 32 and separate learning rates for the backbone and classifier. It ran on one TorchTPU device with bfloat16 autocast. A device-specific BatchNorm workaround was required because the installed TorchTPU build did not update BatchNorm running statistics.

One seed (43) was completed and evaluated for each task. Test AUROC was **0.9476** for Pleural Effusion (18,826 test examples) and **0.9377** for Cardiomegaly (8,205 examples). These are single-run results. They are not estimates of the paper's ten-seed, top-five mean and standard deviation, and the experiment differs from the paper's binary cross-entropy objective by using the repository's two-logit cross-entropy implementation. We therefore describe this as an **adapted reproduction**.

## 1. Introduction

The reference study, *Causal Representation Learning with Observational Grouping for CXR Classification* [1], reports baseline and invariant classification AUROCs for CheXpert and MIMIC-CXR. Its CheXpert tasks include No Finding versus Pleural Effusion and No Finding versus Cardiomegaly. This report documents a conventional, non-invariant ResNet-18 baseline for those two tasks using the CheXpert image mirror available to this environment. The project goal was to establish one usable result per task on TorchTPU, rather than complete the reference paper's full model, invariance, dataset, and seed matrix.

The reference paper describes 224 × 224 inputs, batch size 32, Adam, a backbone learning rate of 0.001, a classifier learning rate of 0.0001, and 20 epochs. It reports ten random seeds and selects the five highest validation-AUROC models for its Table 1 summary. The paper does not name the seed values. Its implementation-details section also describes selection using validation binary cross-entropy, which is not fully consistent with the top-five-by-validation-AUROC description in Section 3.2. We used maximum validation AUROC for the within-run checkpoint and document that choice below.

## 2. Data

### 2.1 Source and provenance

Images and associated metadata came from the [Hugging Face CheXpert dataset](https://huggingface.co/datasets/danjacobellis/chexpert). The downloaded training split contains 223,414 rows in 23 Parquet shards. The records embed both paths/labels and JPEG image bytes. The mirror's small, separate validation split (234 rows) was not used to define this experiment's validation set. Instead, the selected Hugging Face training records were assigned according to the project's existing CheXpert split CSVs.

The repository split files are `datafiles/chexpert/chexpert.sample.train.csv`, `chexpert.sample.val.csv`, and `chexpert.sample.test.csv`. The split is based on the original CheXpert cohort used in the repository. The paper's Table 3 provides class counts but not a public image-ID manifest or enough detail to independently reconstruct every assignment. We verified the counts against Table 3 and verified Hugging Face path, label, and split matches against the local project metadata. All 71,096 selected image paths matched; disease-label values and split membership had zero mismatches. The AP/PA mapping also had zero mismatches. Sex had one metadata mismatch per split and was not used by this baseline.

The preprocessing audit compared 32 Hugging Face images with previously staged copies. None were pixel-identical; mean structural similarity was 0.9467 and the minimum was 0.9247. We therefore used the Hugging Face image bytes consistently for this run. The Hugging Face dataset revision/hash was not recorded in the run config, which limits exact source-version reconstruction.

### 2.2 Cohort construction and labels

For each task, the repository's `CXRDataset` retains records for which exactly one of `No Finding` and the target disease label equals 1. Records where neither is 1 or both are 1 are excluded. The binary target is 1 for the disease and 0 for No Finding. The mirror encodes its source labels as 0 = unlabeled, 1 = uncertain, 2 = absent, and 3 = present; the selected binary CSVs provide the repository's already-derived labels. The cohort filter treats only a value equal to 1 in the selected CSV as a positive finding. In particular, a retained No Finding row whose target label is not 1 receives the negative class; uncertain source labels are not modeled as a third class.

The final examples after cohort filtering are:

| Task | Split | No Finding (negative) | Disease (positive) | Total |
|---|---|---:|---:|---:|
| Pleural Effusion | Train | 6,514 | 31,015 | 37,529 |
| Pleural Effusion | Validation | 1,086 | 5,049 | 6,135 |
| Pleural Effusion | Test | 3,316 | 15,510 | 18,826 |
| Cardiomegaly | Train | 6,514 | 9,353 | 15,867 |
| Cardiomegaly | Validation | 1,086 | 1,548 | 2,634 |
| Cardiomegaly | Test | 3,316 | 4,889 | 8,205 |

These counts match the CheXpert rows in Table 3 of the reference paper [1]. Because the paper does not publish all split image IDs, matching counts alone cannot establish that every image was assigned identically by the original authors. Our matching of split membership is to the available repository split metadata.

### 2.3 Protected attributes

This is a standard baseline, without the paper's invariance objective. The dataset loader maps AP to view code 0 and PA to code 1, and allows both values (`protected_race_set=[0, 1]`). Race and sex are not used for training, sampling, model selection, or reporting.

## 3. Image preprocessing

The preparation script, [`chexpert/prepare_hf_baseline.py`](../chexpert/prepare_hf_baseline.py), performs the following steps:

1. Copies the project's train, validation, and test sample CSVs into `datafiles/chexpert/`.
2. Converts each `path_preproc` into its corresponding Hugging Face CheXpert path.
3. Scans the 23 Parquet shards for the selected records and reads only their embedded image bytes.
4. Decodes each grayscale image, resizes it to 224 × 224 with `skimage.transform.resize(..., preserve_range=True)`, converts the result to `uint8`, and writes PNG output under `chexpert/hf_preprocessed/preproc_224x224/`.
5. Confirms that all 71,096 selected output files exist and records counts and image comparisons in `chexpert/preparation.json`.

The training dataset loader reads the PNG as a float tensor and repeats the grayscale channel three times for the ResNet input. The implementation does **not** scale intensities to [0, 1] or apply ImageNet mean/std normalization. This is the behavior of the repository's `CXRDataset` used here and is a material preprocessing choice to preserve when interpreting or reproducing the result.

Training augmentation follows the repository loader: random horizontal flip with probability 0.5, plus a random affine transform applied with probability 0.5 using rotation up to 15 degrees and scale 0.9–1.1. Validation and test have no augmentation.

## 4. Model and training method

### 4.1 Architecture

The feature extractor is torchvision ResNet-18 initialized with `ResNet18_Weights.IMAGENET1K_V1`, the ImageNet-1K pretrained weights. A grayscale radiograph is repeated across three channels and passed through the standard ResNet-18 stem (7 × 7 convolution, batch normalization, ReLU, and max pooling), then four residual stages with two basic residual blocks per stage and 64, 128, 256, and 512 output channels. Adaptive average pooling produces a 512-dimensional feature vector.

The torchvision 1,000-class `fc` layer is replaced with `nn.Identity`; its ImageNet classifier weights are discarded and do not participate in this experiment. A new `nn.Linear(512, 2)` layer maps the feature vector to two logits, ordered as No Finding and the selected disease. This is instantiated separately for each task and seed.

### 4.1.1 Parameters updated during fine-tuning

This is **full-backbone fine-tuning**, not a frozen-feature-extractor experiment. The code does not set any backbone parameter's `requires_grad` to false. Adam receives all parameters from both the ResNet backbone and the new classifier:

| Component | Parameters adjusted by gradient descent? | Learning rate | Details |
|---|---|---:|---|
| ResNet-18 convolution layers | Yes | 0.001 | Convolution weights throughout the stem and all residual blocks are updated from the CheXpert classification loss. |
| ResNet-18 BatchNorm affine parameters | Yes | 0.001 | Trainable scale (`weight`) and offset (`bias`) parameters are optimized with the backbone. |
| ResNet-18 BatchNorm running buffers | No | N/A | `running_mean`, `running_var`, and `num_batches_tracked` are statistics, not gradient-trained parameters. The TPU hook updates these buffers during training batches. |
| Replaced ImageNet `fc` layer | No | N/A | Removed and replaced by `nn.Identity`, so its original 1,000-class weights are not used or fine-tuned. |
| New two-class linear head | Yes | 0.0001 | Both the 512 × 2 weight matrix and 2-element bias are learned from the task labels. |

For the standard torchvision ResNet-18, the retained backbone has 11,176,512 trainable parameters; the new two-logit classifier has 1,026. Therefore 11,177,538 scalar parameters are exposed to the optimizer per task. The run optimizes the pretrained convolutional and BatchNorm affine values from their ImageNet initialization rather than freezing them. Running-statistic buffers are maintained separately from this parameter count.

### 4.2 Objective and optimization

The reference paper describes a binary classifier trained with binary cross-entropy. The repository's ResNet implementation accepts scalar class labels and uses a two-logit head with `CrossEntropyLoss`; the runner followed that repository behavior. This is a documented loss/head deviation from the paper.

For each task and seed, all backbone and classifier parameters are optimized with Adam using separate parameter groups:

| Parameter group | Learning rate |
|---|---:|
| ResNet-18 backbone | 0.001 |
| Two-logit classifier | 0.0001 |

Training ran for 20 epochs with batch size 32. Training batches are shuffled with a generator initialized from the run seed, and the last incomplete training batch is dropped. Consequently, 37,504 Pleural Effusion examples and 15,840 Cardiomegaly examples contribute to each training epoch. Validation and test use all examples without shuffling. The run sets Python's `random`, NumPy, and PyTorch random seeds to 43 for each task; deterministic kernels were not explicitly enabled.

### 4.3 TPU execution and BatchNorm correction

The run used one TPU device through TorchTPU, with bfloat16 autocast for model forward passes. Loss calculation, score conversion, AUROC, and checkpoint serialization use float32/CPU values as appropriate. The environment versions were Python 3.13, PyTorch `2.15.0+cpu`, torchvision `0.29.2+cpu`, `torch-tpu` 0.2.0, and `libtpu` 0.0.52. Although PyTorch is a CPU wheel, TorchTPU registers a `tpu` device and the model and batches were moved to that device. The runtime logged PJRT API 0.117 for the plugin and 0.116 for the framework; initialization and execution succeeded despite this version warning.

An isolated check showed that the installed TorchTPU BatchNorm implementation used batch statistics during training but did not update its `running_mean` or `running_var` buffers. Without a correction, evaluation used stale buffers and validation AUROC remained near 0.50 while training loss decreased. The final runner installs a forward-pre-hook on all 20 ResNet-18 BatchNorm layers. For each training batch, the hook computes channel mean and unbiased variance across batch/spatial dimensions and applies the PyTorch momentum update (`momentum=0.1`) to the running buffers. A constant-input device probe confirmed the expected mean 0 → 0.5 and variance 1 → 0.9 update. After adding this workaround, the Pleural Effusion validation AUROC rose to approximately 0.94. This device-specific modification is material and should be revalidated if TorchTPU/PyTorch/libtpu versions change; the reported behavior is tracked in TorchTPU issue [3].

### 4.4 Checkpoint selection and metric

After each epoch, the runner computes validation AUROC from the continuous softmax probability for the positive class. The checkpoint with the highest validation AUROC is retained. Once the 20 epochs finish, that checkpoint and its validation predictions are saved. The selected checkpoint is then evaluated on test once. Test AUROC is computed with `sklearn.metrics.roc_auc_score` from continuous positive-class probabilities; no threshold is selected or applied.

The original planned protocol was ten seeds per task followed by top-five test evaluation. The local artifacts contain complete 20-epoch Pleural Effusion histories and checkpoints for seeds 42, 43, and 44; seed 43 has the highest validation AUROC of those three and is the only Pleural Effusion seed included in the final test-evaluated summary. Cardiomegaly seed 43 was the only seed trained for that task. The remaining multi-seed schedule was not completed, so this does not implement the paper's ten-seed/top-five summary protocol.

## 5. Results

### 5.1 Test performance

| Task | Seed | Best epoch | Best validation AUROC | Test AUROC | Test examples |
|---|---:|---:|---:|---:|---:|
| No Finding vs Pleural Effusion | 43 | 12 | 0.9497 | **0.9476** | 18,826 |
| No Finding vs Cardiomegaly | 43 | 19 | 0.9343 | **0.9377** | 8,205 |

The per-task test metrics are individual-run estimates. With one test AUROC per task, a between-seed standard deviation is not estimable; the run summary records it as `null`.

### 5.2 Epoch-level training and validation history

The table reports mean training cross-entropy over the examples used in each epoch and AUROC on the full validation split. Values are rounded to four decimals; the unrounded values are in each seed's `history.json`.

| Epoch | Pleural Effusion train CE | Pleural Effusion val AUROC | Cardiomegaly train CE | Cardiomegaly val AUROC |
|---:|---:|---:|---:|---:|
| 1 | 0.2556 | 0.9253 | 0.4086 | 0.9100 |
| 2 | 0.2240 | 0.9380 | 0.3614 | 0.9231 |
| 3 | 0.2153 | 0.9381 | 0.3468 | 0.9218 |
| 4 | 0.2091 | 0.9436 | 0.3392 | 0.9310 |
| 5 | 0.2041 | 0.9437 | 0.3366 | 0.9266 |
| 6 | 0.2002 | 0.9456 | 0.3292 | 0.9241 |
| 7 | 0.1968 | 0.9455 | 0.3252 | 0.9289 |
| 8 | 0.1929 | 0.9443 | 0.3186 | 0.9309 |
| 9 | 0.1885 | 0.9467 | 0.3159 | 0.9185 |
| 10 | 0.1851 | 0.9425 | 0.3141 | 0.9075 |
| 11 | 0.1853 | 0.9405 | 0.3082 | 0.9287 |
| 12 | 0.1811 | **0.9497** | 0.3050 | 0.9237 |
| 13 | 0.1781 | 0.9465 | 0.2997 | 0.9269 |
| 14 | 0.1743 | 0.9462 | 0.3013 | 0.9246 |
| 15 | 0.1740 | 0.9473 | 0.2889 | 0.9143 |
| 16 | 0.1706 | 0.9482 | 0.2891 | 0.9051 |
| 17 | 0.1668 | 0.9446 | 0.2860 | 0.8981 |
| 18 | 0.1651 | 0.9492 | 0.2804 | 0.9185 |
| 19 | 0.1621 | 0.9450 | 0.2751 | **0.9343** |
| 20 | 0.1595 | 0.9468 | 0.2713 | 0.9320 |

### 5.3 Reference values

For context, the reference paper reports CheXpert ResNet baseline results of 93.04 ± 0.88 AUROC for No Finding versus Pleural Effusion and 89.24 ± 1.07 for No Finding versus Cardiomegaly in Table 1 [1]. Those are top-five-of-ten summaries; our values are single-seed test AUROCs. Given the different loss/head, Hugging Face image bytes, the TPU BatchNorm workaround, and the one-seed scope, the numerical difference should not be interpreted as evidence that one approach is superior.

### 5.4 External TorchXRayVision DenseNet-121 comparison

We also evaluated the pretrained TorchXRayVision DenseNet-121 `densenet121-res224-mimic_ch` checkpoint on the same CheXpert test cohorts. This checkpoint is trained on MIMIC-CXR with CheXpert-labeler targets; its output pathology `Effusion` was mapped to Pleural Effusion, while `Cardiomegaly` maps directly. TorchXRayVision documents this as its 224 × 224 MIMIC-CXR model [5]. We selected MIMIC-CXR pretraining to make this an external-pretraining baseline and avoid the possible test-set overlap associated with a CheXpert-trained checkpoint.

| Task | Examples | ResNet-18 CheXpert seed 43 | XRV DenseNet-121 MIMIC-CXR | XRV minus ResNet-18 |
|---|---:|---:|---:|---:|
| No Finding vs Pleural Effusion | 18,826 | 0.9476 | 0.8892 | -0.0584 |
| No Finding vs Cardiomegaly | 8,205 | 0.9377 | 0.8917 | -0.0460 |

Both models were scored on the same project test CSV and the same task-specific binary cohort filters. For TorchXRayVision, grayscale pixels were normalized from 8-bit values to its expected `[-1024, 1024]` range; the stored images were already square 224 × 224, so no further crop or resize was needed. The ResNet-18 input path instead uses the repository's pseudo-RGB images without this intensity normalization. The comparison therefore measures each model under its expected input scaling; it does not isolate the architecture effect.

The TorchXRayVision model ran in TorchTPU compiled mode with bfloat16 autocast and batch size 64. Its output operating-point calibration was disabled and raw pathology logits were used. The calibration applies a monotonic transformation per output, which preserves AUROC rankings; no threshold was selected on this test set. The evaluator, configuration (including the XRV weight SHA-256), and per-example scores are saved under the ignored local path `chexpert/runs/resnet18_baseline_tpu_ce/torchxrayvision_densenet121_mimic_ch/`. Reproduce it with TorchXRayVision 1.5.7 installed and:

```bash
python -m pip install torchxrayvision==1.5.7
env \
  PYTHONPATH="$PWD" \
  CXR_DEVICE=tpu \
  CXR_TPU_AMP=1 \
  CXR_TPU_COMPILE=1 \
  CXR_NUM_WORKERS=4 \
  CXR_EVAL_BATCH_SIZE=64 \
  python scripts/evaluate_torchxrayvision_chexpert.py
```

On these test cohorts, the fine-tuned ResNet-18 scored 0.0584 AUROC higher for Pleural Effusion and 0.0460 higher for Cardiomegaly than this external pretrained DenseNet-121. These are descriptive differences between two differently pretrained and differently processed models, without confidence intervals or a multi-seed analysis; they should not be attributed to architecture alone.

### 5.5 TorchXRayVision DenseNet-121 pretrained on CheXpert

At the user's request, we also evaluated TorchXRayVision's `densenet121-res224-chex` checkpoint, whose documented training source is CheXpert. Evaluation used the same held-out project CSVs, binary cohort filters, TorchTPU compiled execution, bfloat16 autocast, and batch size 64 as the MIMIC-CXR-pretrained comparison. This is a higher-overlap-risk comparison: the checkpoint may have trained on some or all images in our test cohorts. TorchXRayVision's precise training image list and split were not available for verifying overlap, so treat these scores as exploratory and potentially non-independent; they are not an unbiased test-set comparison.

| Task | Examples | ResNet-18 CheXpert seed 43 | XRV DenseNet-121 CheXpert | XRV minus ResNet-18 |
|---|---:|---:|---:|---:|
| No Finding vs Pleural Effusion | 18,826 | 0.9476 | 0.9320 | -0.0156 |
| No Finding vs Cardiomegaly | 8,205 | 0.9377 | 0.9330 | -0.0047 |

The XRV model used its expected grayscale normalization to `[-1024, 1024]`; inputs were already 224 × 224. Raw pathology logits were used for AUROC, with XRV sigmoid/operating-point calibration disabled because the per-output monotonic transform does not change score ranking. Weight identifier: `densenet121-res224-chex`; SHA-256: `1a914ae802170187a4852e594708206c12468eeeefd66aa4e8e26141649d61a4`. The evaluator, configuration, and per-example predictions are in the ignored local directory `chexpert/runs/resnet18_baseline_tpu_ce/torchxrayvision_densenet121_chex/`.

Both AUROCs are below the corresponding ResNet-18 values, by 0.0156 and 0.0047. Because the pretrained checkpoint may have seen test images and differs in training data, architecture, and preprocessing, these differences are descriptive only and do not establish comparative generalization. The evaluation metrics and prediction files were written successfully; the process then encountered a segmentation fault during compiled TorchTPU executable cleanup.

## 6. Reproducibility and artifacts

The runner is [`chexpert/run_resnet18_baseline.py`](../chexpert/run_resnet18_baseline.py). The preprocessing code and broader protocol notes are [`chexpert/prepare_hf_baseline.py`](../chexpert/prepare_hf_baseline.py) and [`docs/chexpert-table1-training-plan.md`](chexpert-table1-training-plan.md). The run used:

```bash
env \
  PYTHONPATH="$PWD" \
  CXR_DEVICE=tpu \
  CXR_TPU_AMP=1 \
  CXR_NUM_WORKERS=4 \
  CXR_SEEDS=43 \
  CXR_TASKS='Pleural Effusion,Cardiomegaly' \
  python \
  chexpert/run_resnet18_baseline.py
```

The machine-readable results are in the ignored local directory `chexpert/runs/resnet18_baseline_tpu_ce/`:

- `config.json`: data source, tasks, seed, hyperparameters, TPU/AMP settings, and deviations.
- `pleural_effusion/seed_43/` and `cardiomegaly/seed_43/`: best checkpoints, per-epoch histories, and validation/test prediction CSVs.
- `pleural_effusion/summary.json`, `cardiomegaly/summary.json`, and root `summary.json`: validation/test metrics.

The images and intermediate CSVs live under git-ignored local paths (`chexpert/` and `datafiles/chexpert/`) and are not included in this report. The dataset revision hash, exact source file hashes, git commit hash, a full stdout log file, and TPU model identifier were not captured in the run metadata. These are remaining reproducibility gaps. The paper's split counts were matched, but its exact image-level assignments cannot be independently authenticated from the published counts alone.

## 7. Limitations and conclusion

This experiment establishes one completed TPU-trained ResNet-18 result for each binary CheXpert task. It uses one seed per task, so it provides no estimate of seed variance and does not reproduce the paper's top-five-of-ten aggregation. Additional material deviations include using the Hugging Face image mirror rather than the original Stanford distribution, a repository two-logit cross-entropy head instead of the paper's stated BCE objective, and a manual BatchNorm-statistics hook required by the installed TorchTPU build. The exact source revision and several provenance hashes were not saved. AUROC was calculated from continuous scores on a test split whose published class counts match the paper, but exact original image membership is not verifiable from the paper.

Accordingly, report the two test AUROCs as **single-seed adapted-reproduction results**: 0.9476 for Pleural Effusion and 0.9377 for Cardiomegaly. Do not present them as a ten-seed mean, claim an exact reproduction, or infer an improvement over the paper's reported averages.

## References

1. Rasal, R., Kori, A., and Glocker, B. “Causal Representation Learning with Observational Grouping for CXR Classification.” arXiv:2506.20582, 2025. [https://arxiv.org/abs/2506.20582](https://arxiv.org/abs/2506.20582).
2. Hugging Face dataset repository, `danjacobellis/chexpert`. [https://huggingface.co/datasets/danjacobellis/chexpert](https://huggingface.co/datasets/danjacobellis/chexpert).
3. Google PyTorch TorchTPU issue tracker, “BatchNorm never updates running_mean/running_var in training mode.” [https://github.com/google-pytorch/torch_tpu/issues/6465](https://github.com/google-pytorch/torch_tpu/issues/6465).
4. TorchTPU API reference, automatic mixed precision. [https://torchtpu.readthedocs.io/en/latest/api_reference/](https://torchtpu.readthedocs.io/en/latest/api_reference/).
5. TorchXRayVision, model definitions and pretrained weights, including `densenet121-res224-mimic_ch`. [https://github.com/mlmed/torchxrayvision/blob/main/torchxrayvision/models.py](https://github.com/mlmed/torchxrayvision/blob/main/torchxrayvision/models.py).
