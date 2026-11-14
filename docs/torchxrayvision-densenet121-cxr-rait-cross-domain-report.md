# TorchXRayVision DenseNet-121 Evaluation on CXR-RAIT

## Summary

We evaluated two pretrained TorchXRayVision DenseNet-121 checkpoints on the same 3,248-image CXR-RAIT cohort used in the [CheXpert ResNet-18 cross-domain evaluation](resnet18-chexpert-cxr-rait-cross-domain-report.md). The checkpoints were trained on CheXpert and MIMIC-CXR, respectively. Neither was fine-tuned or calibrated on CXR-RAIT.

| Pretraining source | Task | Examples | Positive | Negative | AUROC | Average precision |
|---|---|---:|---:|---:|---:|---:|
| CheXpert | Pleural Effusion | 3,248 | 534 | 2,714 | **0.7591** | 0.4232 |
| CheXpert | Cardiomegaly | 3,248 | 567 | 2,681 | **0.7952** | 0.4481 |
| MIMIC-CXR (CheXpert labels) | Pleural Effusion | 3,248 | 534 | 2,714 | **0.7289** | 0.4128 |
| MIMIC-CXR (CheXpert labels) | Cardiomegaly | 3,248 | 567 | 2,681 | **0.7076** | 0.3012 |

For context, the CXR-RAIT ResNet-18 checkpoints previously evaluated on this cohort achieved AUROC 0.8499 for Pleural Effusion and 0.8464 for Cardiomegaly. These are descriptive comparisons only: the models differ in architecture, pretraining, and input normalization. The CheXpert-pretrained DenseNet checkpoint's exact training image list was not checked for overlap with this target cohort.

## Data and labels

The evaluation used `all_years_rekonsiliasi_final_pleural_effusion_cardiomegaly.csv`, copied from `gs://cxr-rait/rait-drive-source/`. All 3,248 rows were evaluated for both tasks. The CSV SHA-256 is `0499eaebeb7c2221f1aa3ba8b32c1b6e4b9238ca6d1effe324ac7d01e7cade18`.

The binary labels were supplied in the CSV and thresholded from `prediction_nlp` at 0.5. A target of 0 indicates that the target disease score was below the threshold; it serves as the negative or “No Finding” proxy and is not confirmation of a normal radiograph. These metrics therefore measure agreement with NLP-derived labels, not independently adjudicated clinical ground truth.

## Models and evaluation method

| Training source | TorchXRayVision weights | Checkpoint SHA-256 |
|---|---|---|
| CheXpert (Stanford) | `densenet121-res224-chex` | `1a914ae802170187a4852e594708206c12468eeeefd66aa4e8e26141649d61a4` |
| MIMIC-CXR with CheXpert-labeler targets | `densenet121-res224-mimic_ch` | `23b13a04459684ffa41247d068207a7657b4e1b4dec2b02431f5026bd75b1189` |

For each task, the XRV `Effusion` or `Cardiomegaly` output was compared with the corresponding CXR-RAIT binary label. Scores were raw model logits; XRV's sigmoid and operating-point calibration were omitted because AUROC depends only on ranking. Average precision was also computed directly from those raw scores, which is invariant to a monotonic sigmoid transform.

The same pre-rendered 224 × 224 CXR-RAIT PNGs were used for both checkpoints. These were produced from the matched DICOMs using modality LUT, VOI LUT/window when present, `MONOCHROME1` inversion, min-max 8-bit scaling, and resize to 224 × 224. Images were converted to grayscale and normalized with XRV's `normalize(maxval=255)` to its expected `[-1024, 1024]` input range. Inference ran on CPU, batch size 32, using TorchXRayVision 1.5.7. No target-domain threshold was tuned.

## Artifacts and reproduction

The evaluator is [`scripts/evaluate_torchxrayvision_rait.py`](../scripts/evaluate_torchxrayvision_rait.py). Machine-readable metrics and per-image scores are stored in the ignored runtime directory `chexpert/runs/resnet18_baseline_tpu_ce/cxr_rait/xrv_results/`.

Run from the repository root with the project's TorchXRayVision environment:

```bash
python scripts/evaluate_torchxrayvision_rait.py \
  --device cpu \
  --batch-size 32 \
  --num-workers 0
```

The run used the already cached checkpoint files from the CheXpert experiment directories; no weights were downloaded. Summary JSON is at `chexpert/runs/resnet18_baseline_tpu_ce/cxr_rait/xrv_results/summary.json`.
