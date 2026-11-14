# CheXpert ResNet-18 Cross-Domain Evaluation on CXR-RAIT

## Summary

The two existing seed-43 ResNet-18 classifiers trained on CheXpert were evaluated on 3,248 CXR-RAIT images. Pleural Effusion achieved AUROC **0.8499** and Cardiomegaly achieved AUROC **0.8464** against the binary labels in the CXR-RAIT labeled CSV.

| Task | Examples | Positive | Negative | AUROC | Average precision |
|---|---:|---:|---:|---:|---:|
| Pleural Effusion | 3,248 | 534 | 2,714 | **0.8499** | 0.5710 |
| Cardiomegaly | 3,248 | 567 | 2,681 | **0.8464** | 0.5585 |

Scores are positive-class softmax probabilities. No decision threshold was selected on CXR-RAIT. These are single-checkpoint point estimates; confidence intervals and multi-seed variation were not calculated.

## Data and label definition

The evaluation used `all_years_rekonsiliasi_final_pleural_effusion_cardiomegaly.csv` from `gs://cxr-rait/rait-drive-source/`. Its 3,248 `cv_filename` values all matched DICOM files in that bucket. The CSV labels are the supplied binary labels thresholded from `prediction_nlp` at 0.5. The evaluation used all 3,248 rows independently for each task.

For this experiment, label `1` is the disease-positive class and label `0` is the negative class used by the corresponding CheXpert binary classifier. The CSV does not include a `normal`/`No Finding` label. Thus, label `0` means the target disease score was below 0.5; it is a **No Finding proxy**, not confirmation that the radiograph is normal. The scores measure agreement with NLP-derived labels rather than performance against independently adjudicated clinical ground truth.

The supplied CSV SHA-256 is `0499eaebeb7c2221f1aa3ba8b32c1b6e4b9238ca6d1effe324ac7d01e7cade18`.

## Models and preprocessing

Each task used its own seed-43 CheXpert ResNet-18 checkpoint, selected during the original CheXpert run by validation AUROC. No model was retrained, fine-tuned, or calibrated on CXR-RAIT.

| Task | Checkpoint | SHA-256 |
|---|---|---|
| Pleural Effusion | `chexpert/runs/resnet18_baseline_tpu_ce/pleural_effusion/seed_43/best.pt` | `a865732bb4696b812d10e6fde8fd3ff481517fabca02772ac17c80c245fd8bce` |
| Cardiomegaly | `chexpert/runs/resnet18_baseline_tpu_ce/cardiomegaly/seed_43/best.pt` | `a318a978827b4a5cf01194764724b412d835155701505fa92df3b1d348b2f431` |

DICOM pixels were decoded with pydicom 3.0.2. The preprocessing applied the modality LUT and any VOI LUT/window metadata, inverted `MONOCHROME1` images, scaled the rendered grayscale values to 8-bit, resized to 224 × 224, and repeated the grayscale image across three channels. As in the CheXpert training pipeline, inputs remained on the 0–255 scale without ImageNet mean/std normalization.

The run used CPU inference because TorchTPU did not report an available device in this session. The CheXpert internal test AUROCs were 0.9476 for Pleural Effusion and 0.9377 for Cardiomegaly. Those values are included only as context: the internal and external evaluations use different datasets, target-label sources, and image preprocessing, so their difference is not an isolated estimate of domain shift.

## Artifacts and reproduction

The evaluator is [`scripts/evaluate_resnet18_rait.py`](../scripts/evaluate_resnet18_rait.py). The machine-readable summary and per-image predictions are under the ignored runtime directory `chexpert/runs/resnet18_baseline_tpu_ce/cxr_rait/results/`. The DICOMs and 224 × 224 preprocessed PNGs remain in ignored runtime directories and are not included in Git.

The run was invoked with:

```bash
PYTHONPATH=/tmp/rait_pydicom:$PWD \
CXR_DEVICE=cpu \
python scripts/evaluate_resnet18_rait.py \
  --num-workers 0 \
  --preprocess-workers 4
```

The run-specific pydicom installation was kept under `/tmp/rait_pydicom`; no project dependency files were changed.
