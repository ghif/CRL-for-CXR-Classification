# Cross-Domain Evaluation of CheXpert and TorchXRayVision Models on CXR-RAIT

**Report date:** 14 November 2026  
**Study type:** Retrospective, zero-shot cross-domain evaluation with CPU and TPU inference

## Abstract

### Background

Chest radiograph classifiers are often evaluated outside the dataset used for training, but performance can vary with image acquisition, patient population, preprocessing, and label construction. We assessed three pretrained model configurations on the same CXR-RAIT target cohort: task-specific ResNet-18 classifiers trained on CheXpert, and TorchXRayVision (XRV) DenseNet-121 checkpoints pretrained on CheXpert or MIMIC-CXR.

### Methods

The target cohort comprised 3,248 images with binary labels for pleural effusion and cardiomegaly. Labels were thresholded from the supplied `prediction_nlp` values at 0.5. No model was fine-tuned or calibrated on CXR-RAIT. Discrimination was summarized using AUROC and average precision (AP).

### Results

On TPU, pleural-effusion AUROC/AP were 0.8501/0.5705 for CheXpert ResNet-18, 0.7591/0.4229 for CheXpert-pretrained XRV DenseNet-121, and 0.7290/0.4121 for MIMIC-CXR-pretrained XRV DenseNet-121. Cardiomegaly AUROC/AP were 0.8465/0.5585, 0.7952/0.4480, and 0.7079/0.3015, respectively. TPU estimates were very close to the earlier CPU estimates.

### Conclusions

All three configurations showed above-chance ranking against the supplied CXR-RAIT labels. The CheXpert ResNet-18 checkpoints had the highest point-estimate AUROC and AP on both tasks. Since the comparisons vary in architecture, model training procedure, and input normalization, they do not isolate the effect of training domain or architecture. Labels are NLP-derived and the target-negative class is not a verified normal cohort; results should be interpreted as exploratory cross-domain agreement estimates.

**Keywords:** chest radiography; domain shift; pleural effusion; cardiomegaly; external evaluation; TorchXRayVision

## 1. Introduction

Models trained on one chest radiograph dataset may encounter a different distribution when applied to images from another institution or data collection. Cross-domain evaluation can reveal whether a model's ranking behavior is retained, but the measured performance depends on both the image pipeline and the target labels. Here, we report a common-cohort comparison of three existing model configurations on CXR-RAIT: a CheXpert-trained ResNet-18 for each target finding, and two pretrained XRV DenseNet-121 models whose training sources are CheXpert and MIMIC-CXR.

The evaluation is zero-shot: model parameters and operating thresholds were not adapted using CXR-RAIT. The purpose is descriptive comparison against the available binary labels, not clinical validation or a controlled model ablation.

## 2. Methods

### 2.1 Target cohort and labels

The cohort was drawn from `all_years_rekonsiliasi_final_pleural_effusion_cardiomegaly.csv` in `gs://cxr-rait/rait-drive-source/`. The CSV contains 3,248 unique `cv_filename` values; these were previously matched to DICOM objects in the bucket. All 3,248 rows were evaluated for each task. The CSV SHA-256 is:

```text
0499eaebeb7c2221f1aa3ba8b32c1b6e4b9238ca6d1effe324ac7d01e7cade18
```

The supplied target labels are binary values thresholded from `prediction_nlp` at 0.5. Pleural effusion had 534 positives and 2,714 negatives; cardiomegaly had 567 positives and 2,681 negatives. In this report, label 0 is referred to as the target-negative class. It can be viewed as a “No Finding” proxy for the specific target label, but it does not establish that an image is clinically normal or free of other findings. Consequently, measured performance is agreement with NLP-derived labels rather than independently adjudicated clinical ground truth.

### 2.2 Evaluated models

| Configuration | Model and training source | Target outputs | Checkpoint provenance |
|---|---|---|---|
| CheXpert ResNet-18 | Task-specific ResNet-18 classifiers trained on CheXpert | Pleural Effusion; Cardiomegaly | Seed 43; selected using source-domain validation AUROC |
| XRV DenseNet-121, CheXpert | TorchXRayVision `densenet121-res224-chex`, trained on CheXpert | `Effusion`; `Cardiomegaly` | TorchXRayVision 1.5.7 pretrained checkpoint |
| XRV DenseNet-121, MIMIC-CXR | TorchXRayVision `densenet121-res224-mimic_ch`, trained on MIMIC-CXR with CheXpert-labeler targets | `Effusion`; `Cardiomegaly` | TorchXRayVision 1.5.7 pretrained checkpoint |

The ResNet-18 checkpoints were loaded independently for the two tasks. The two DenseNet results came from the corresponding outputs of each multi-label XRV checkpoint. No weights were updated on the target cohort.

Checkpoint SHA-256 fingerprints:

| Configuration | Task / weight | SHA-256 |
|---|---|---|
| CheXpert ResNet-18 | Pleural Effusion, seed 43 | `a865732bb4696b812d10e6fde8fd3ff481517fabca02772ac17c80c245fd8bce` |
| CheXpert ResNet-18 | Cardiomegaly, seed 43 | `a318a978827b4a5cf01194764724b412d835155701505fa92df3b1d348b2f431` |
| XRV DenseNet-121, CheXpert | `densenet121-res224-chex` | `1a914ae802170187a4852e594708206c12468eeeefd66aa4e8e26141649d61a4` |
| XRV DenseNet-121, MIMIC-CXR | `densenet121-res224-mimic_ch` | `23b13a04459684ffa41247d068207a7657b4e1b4dec2b02431f5026bd75b1189` |

### 2.3 Image processing and inference

DICOM images were rendered using pydicom 3.0.2. The shared rendering step applied the modality LUT, then the VOI LUT or window information when present, inverted `MONOCHROME1`, scaled image values to 8-bit using per-image min-max scaling, and resized to 224 × 224 pixels.

The model input transformations then differed by model family, following their respective pipelines:

* **ResNet-18:** grayscale was repeated across three channels and retained on the 0–255 scale, without ImageNet mean or standard deviation normalization.
* **XRV DenseNet-121:** grayscale was normalized using XRV's `normalize(maxval=255)` to its expected `[-1024, 1024]` range.

Each model configuration was evaluated first on CPU and then on a TPU v6e using batch size 32. The TPU runs used TorchTPU 0.2.0 and libtpu 0.0.52. ResNet-18 inference used bfloat16 autocast on TPU without `torch.compile`. XRV inference used bfloat16 autocast and fixed-shape `torch.compile` (`backend="tpu"`, `dynamic=False`); its final partial batch was padded to 32 and the padded outputs were discarded. ResNet-18 scores were positive-class softmax probabilities. XRV scores were raw pathology logits; sigmoid/operating-point calibration was omitted. These score transformations preserve ranking and therefore do not change AUROC or average precision. No decision threshold was selected on the target domain.

### 2.4 Metrics

For each task and model configuration, we computed AUROC and average precision over all target-cohort rows. These are single-checkpoint point estimates. No confidence intervals, bootstrap intervals, multi-seed estimates, sensitivity, specificity, or threshold-based accuracy were calculated.

## 3. Results

### 3.1 Cross-domain performance

| Model configuration | Task | N | Positive | Negative | CPU AUROC | CPU AP | TPU AUROC | TPU AP |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| CheXpert ResNet-18 | Pleural Effusion | 3,248 | 534 | 2,714 | 0.8499 | 0.5710 | **0.8501** | **0.5705** |
| XRV DenseNet-121, CheXpert | Pleural Effusion (`Effusion`) | 3,248 | 534 | 2,714 | 0.7591 | 0.4232 | 0.7591 | 0.4229 |
| XRV DenseNet-121, MIMIC-CXR | Pleural Effusion (`Effusion`) | 3,248 | 534 | 2,714 | 0.7289 | 0.4128 | 0.7290 | 0.4121 |
| CheXpert ResNet-18 | Cardiomegaly | 3,248 | 567 | 2,681 | 0.8464 | 0.5585 | **0.8465** | **0.5585** |
| XRV DenseNet-121, CheXpert | Cardiomegaly | 3,248 | 567 | 2,681 | 0.7952 | 0.4481 | 0.7952 | 0.4480 |
| XRV DenseNet-121, MIMIC-CXR | Cardiomegaly | 3,248 | 567 | 2,681 | 0.7076 | 0.3012 | 0.7079 | 0.3015 |

The TPU runs reproduce the same ordering as the CPU runs: CheXpert ResNet-18 had the highest point-estimate AUROC and AP for both tasks, while the CheXpert-pretrained XRV checkpoint exceeded the MIMIC-CXR-pretrained XRV checkpoint on both tasks. CPU-to-TPU differences were small (at most 0.0003 AUROC and 0.0007 AP at the reported precision), consistent with the bfloat16 TPU path preserving score rankings closely. Because uncertainty intervals were not estimated, between-model differences are descriptive and do not establish statistical significance.

### 3.2 Source-domain context

The seed-43 CheXpert ResNet-18 checkpoints had source-domain test AUROC of 0.9476 for Pleural Effusion and 0.9377 for Cardiomegaly. Those values come from a separate CheXpert test cohort and its labels. They are provided as context only and are not directly comparable estimates of domain shift because the datasets, target-label construction, and image pipelines differ.

## 4. Discussion

In this cohort, all three evaluated configurations ranked examples above chance relative to the binary CXR-RAIT labels. The TPU rerun preserved the CPU result pattern, with only small numerical differences under bfloat16 inference. The two task-specific CheXpert ResNet-18 checkpoints yielded the strongest point estimates. Both XRV DenseNet checkpoints also retained discriminatory signal, though their scores were lower on this cohort. Across the XRV checkpoints, the CheXpert-pretrained model had higher point estimates than the MIMIC-CXR-pretrained model for both findings.

These patterns should not be attributed solely to the source domain. The ResNet-18 and DenseNet-121 differ in architecture, training recipes, output parameterization, and expected image scaling. The XRV CheXpert and MIMIC-CXR checkpoints also may differ in more than their training image source, including cohort composition and label generation. In addition, the exact training-image lists for these public checkpoints were not compared against CXR-RAIT; possible overlap was not assessed.

The observed AUROC and AP values quantify association with the supplied thresholded NLP labels. Differences can reflect image-domain shift, label noise, prevalence and spectrum differences, rendering choices, or combinations of these factors. These evaluations do not demonstrate clinical utility, establish calibrated probabilities, or support a deployment threshold.

## 5. Limitations

1. **Reference-label limitations:** CXR-RAIT targets are NLP-derived `prediction_nlp` scores thresholded at 0.5, not independently adjudicated labels. A zero target is not proof of a normal radiograph.
2. **No uncertainty estimates:** results are point estimates from one checkpoint per configuration. No confidence intervals or seed variability were assessed.
3. **Non-controlled model comparisons:** architecture, pretraining recipe, checkpoint selection, and model-specific input scaling differ. The observed ranking does not isolate the effect of architecture or pretraining dataset.
4. **Possible data overlap:** checkpoint training-image membership was not checked against CXR-RAIT, especially for the CheXpert-pretrained weights.
5. **Preprocessing dependence:** all inputs were rendered with per-image min-max scaling from DICOM. Alternative windowing or normalization may change performance; the ResNet and XRV input tensors also followed different normalization conventions.
6. **Limited scope:** evaluation covered two labels and one target cohort and did not measure calibration or threshold-dependent clinical operating characteristics.

## 6. Conclusion

On 3,248 CXR-RAIT images, the CheXpert-trained ResNet-18 checkpoints achieved the highest observed AUROC and AP for pleural effusion and cardiomegaly, followed by the CheXpert-pretrained XRV DenseNet-121 and then the MIMIC-CXR-pretrained XRV DenseNet-121. Given the weak target-label reference standard, lack of uncertainty estimates, and non-controlled model and preprocessing differences, the results are best treated as an exploratory benchmark. A stronger follow-up study would use independently reviewed target labels, quantify uncertainty, assess data overlap, and standardize or explicitly ablate image preprocessing and model selection.

## 7. Reproducibility and artifacts

The evaluation scripts are [`evaluate_resnet18_rait.py`](../scripts/evaluate_resnet18_rait.py) and [`evaluate_torchxrayvision_rait.py`](../scripts/evaluate_torchxrayvision_rait.py). The machine-readable summaries and per-image predictions are stored in ignored runtime directories:

* CPU ResNet-18: `chexpert/runs/resnet18_baseline_tpu_ce/cxr_rait/results/`
* TPU ResNet-18: `chexpert/runs/resnet18_baseline_tpu_ce/cxr_rait/results_tpu/`
* CPU XRV DenseNet-121: `chexpert/runs/resnet18_baseline_tpu_ce/cxr_rait/xrv_results/`
* TPU XRV DenseNet-121: `chexpert/runs/resnet18_baseline_tpu_ce/cxr_rait/xrv_results_tpu/`

Both scripts ran in the `crl-cxr` environment with batch size 32. The ResNet evaluation used pydicom 3.0.2 and the repository's CheXpert ResNet preprocessing. The XRV evaluation used TorchXRayVision 1.5.7 and local cached checkpoints. The TPU artifacts record `device: tpu`, bfloat16 inference, and XRV compilation status in their JSON summaries. Example TPU commands from the repository root:

```bash
PYTHONPATH=/tmp/rait_pydicom:$PWD \
CXR_DEVICE=tpu \
python scripts/evaluate_resnet18_rait.py \
  --device tpu \
  --batch-size 32 \
  --num-workers 0 \
  --preprocess-workers 4 \
  --output-root chexpert/runs/resnet18_baseline_tpu_ce/cxr_rait/results_tpu

python scripts/evaluate_torchxrayvision_rait.py \
  --device tpu \
  --batch-size 32 \
  --num-workers 0 \
  --compile-model \
  --output-root chexpert/runs/resnet18_baseline_tpu_ce/cxr_rait/xrv_results_tpu
```

The CXR-RAIT images and patient-level or study-level metadata are not included in this repository.
