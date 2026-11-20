# Cross-Domain Chest X-Ray Classification and Test-Time Adaptation with Pretrained DenseNet-121

## Abstract

We present a systematic cross-domain evaluation of four pretrained TorchXRayVision DenseNet-121 classifiers—trained on **CheXpert**, **PadChest**, **MIMIC-CXR**, and **NIH ChestX-ray14**—across **six target cohorts** (**CheXpert**, **PadChest**, **CXR-RAIT**, **MIMIC-CXR**, **X-Raydar**, and **NIH ChestX-ray14**, totaling **63,791** test radiographs) on **TorchTPU**. Across all **24 source–target pairs** (20 cross-domain and 4 in-domain reference baselines), we evaluate both **unadapted zero-shot transfer (`No Adapt.`)** and **Co-occurrence Weighted Adaptation (`CoWA`)**, a source-free test-time adaptation (TTA) method proposed at MICCAI 2026.

Our experiments reveal three main findings:
1. **PadChest generalizes best across external domains:** Without adaptation, `densenet121-res224-pc` achieves the highest cross-domain macro AUROC on **NIH ChestX-ray14** (`0.7126`, exceeding even the in-domain NIH model at `0.7025`), **X-Raydar** (`0.7527`), and **CXR-RAIT** (`0.7000`).
2. **Exact alignment with published benchmarks:** On the official NIH `test_list.txt` split, our `CheXpert → NIH` unadapted baseline (`0.6457` macro AUROC) matches Table 1 of Jeong et al. (MICCAI 2026, `64.6`) to within `0.0003`.
3. **CoWA TTA sensitivity to stream length and logit calibration:** With fixed paper-default hyperparameters (`lr = 1e-3`, `w_min = 0.01`) and uncalibrated `sigmoid(raw_logits)` (required to avoid `aten.index_put` on TorchTPU), CoWA improves short-stream transfer on `CheXpert → X-Raydar` (`0.6727 → 0.6921`, `+0.0193` over 16 batches) but degrades performance on long target streams (`174–400` batches on PadChest, CheXpert, and NIH) due to cumulative BatchNorm affine drift.

---

## 1. Experimental Setup

### 1.1 Source Models and Shared Pathology Space

All source models are `torchxrayvision==1.5.7` DenseNet-121 backbones (`op_threshs = None`, `apply_sigmoid = False` for TorchTPU compatibility):

| Source Key | TorchXRayVision Weights | Checkpoint SHA-256 |
| --- | --- | --- |
| `chexpert` | `densenet121-res224-chex` | `1a914ae802170187a4852e594708206c12468eeeefd66aa4e8e26141649d61a4` |
| `padchest` | `densenet121-res224-pc` | `a9148ef62ae4e7a31a9bed8cef22811f39681cd61e4390398936efc6280af521` |
| `mimic_ch` | `densenet121-res224-mimic_ch` | `23b13a04459684ffa41247d068207a7657b4e1b4dec2b02431f5026bd75b1189` |
| `nih` | `densenet121-res224-nih` | `31a5475c6b1383a7edbfcaf96af9ce302eb9969cfe28b6b8b5859afc6f274fdf` |

We evaluate **6 shared thoracic pathologies**: `Atelectasis`, `Cardiomegaly`, `Pleural Effusion` (`Effusion` head), `Consolidation`, `Pneumothorax`, and `Edema`. On **CXR-RAIT** and **X-Raydar**, `Edema` is omitted because no direct `edema` label exists in their annotation schemas; we therefore report both **5-pathology** (common to all 6 targets) and **6-pathology** macro averages.

### 1.2 Target Cohorts and Preprocessing

| Target Cohort | N | Label Source & Binarization | Image Format & Normalization |
| --- | ---: | --- | --- |
| **CheXpert** | 21,438 | `chexpert.sample.test.csv` joined with HF `danjacobellis/chexpert` Parquet shards; U-Zeroes (`code == 3` → `1`) | 224×224 `uint8` PNG, `normalize(maxval=255)` |
| **PadChest** | 11,111 | `padchest-stage-manifest.jsonl` (`split == "test"`, frontal `PA`/`AP`/`AP_horizontal`) + `PADCHEST_..._160K_01.02.19.csv` | 224×224 **`uint16`** PNG, `normalize(maxval=65535)` |
| **CXR-RAIT** | 3,248 | `labels_5pathologies.csv` (`prediction_nlp >= 0.5` from `all_years_rekonsiliasi_final.csv`); 5 pathologies | 224×224 `uint8` PNG, `normalize(maxval=255)` |
| **MIMIC-CXR** | 1,419 | Frontal `AP`/`PA` from Kaggle `simhadrisadaram/mimic-cxr-dataset` (`mimic_cxr_aug_validate.csv`), negation-aware regex over `text` | 512×512 RGB JPEG → grayscale, `normalize(255)`, `XRayResizer(224)` |
| **X-Raydar** | 979 | `annotations.jsonl` radiologist-adjudicated `consensus_labels`; 5 pathologies | Full-res `uint8` PNG → `normalize(255)`, `XRayResizer(224)` |
| **NIH ChestX-ray14** | 25,596 | Official `test_list.txt` + `Data_Entry_2017.csv` from Kaggle `nih-chest-xrays/data`, pipe-delimited `Finding Labels` | 1024×1024 `uint8` PNG → `normalize(255)`, `XRayResizer(224)` |

### 1.3 Hardware, Environment, and CoWA Protocol

All runs executed on **TorchTPU** (`/mnt/data/miniconda/envs/crl-cxr`, Python 3.13, `torch-tpu==0.2.0`, `bfloat16` autocast, batch size 64).
- **Unadapted (`No Adapt.`):** Raw logits evaluated in `eval()` mode; compiled runs (`torch.compile(backend="tpu", dynamic=False)`) achieved up to **2,871 img/s** on pre-resized 224×224 cohorts.
- **CoWA TTA (`CoWA`):** Ported from [`woojin716/CoWA`](https://github.com/woojin716/CoWA) to TorchTPU (`evaluate_cowa_cross_domain.py`). Because TorchTPU cannot trace `op_threshs` (`aten.index_put`), probabilities for entropy and co-occurrence estimation are computed via `torch.sigmoid(raw_logits)`. Only BatchNorm affine parameters (`weight`, `bias`) are updated using Adam (`lr = 1e-3`, `steps = 1`, `cooccur_threshold = 0.5`, `pattern_temp = 0.1`, `min_weight = 0.01`, `seed = 42`), with a `train()`-mode BatchNorm EMA hook (`install_tpu_batchnorm_stats_fix`) so Pass 2 evaluation uses updated running statistics.

---

## 2. Main Results: Unadapted (`No Adapt.`) vs. CoWA TTA

Table 1 reports the complete 4×6 evaluation matrix across all **24 pairs** (macro AUROC and macro AUPRC over all available pathologies per target, plus the 5-pathology common macro AUROC).

| Source → Target | Regime | N | 5-Path AUROC (`No Adapt.` → `CoWA`) | All-Path AUROC (`No Adapt.` → `CoWA`) | Δ AUROC | All-Path AUPRC (`No Adapt.` → `CoWA`) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `chexpert → chexpert` | In-domain | 21,438 | 0.7121 → 0.6228 | 0.7173 → 0.6314 | −0.0859 | 0.3736 → 0.3281 |
| `padchest → chexpert` | Cross-domain | 21,438 | **0.6792** → 0.5503 | **0.6858** → 0.5641 | −0.1217 | **0.3673** → 0.2926 |
| `mimic_ch → chexpert` | Cross-domain | 21,438 | 0.6534 → 0.5406 | 0.6665 → 0.5586 | −0.1079 | 0.3378 → 0.2889 |
| `nih → chexpert` | Cross-domain | 21,438 | 0.6458 → 0.5526 | 0.6503 → 0.5533 | −0.0970 | 0.3369 → 0.2826 |
| `chexpert → padchest` | Cross-domain | 11,111 | 0.7487 → 0.6091 | 0.7747 → 0.6024 | −0.1723 | 0.1387 → 0.0856 |
| `padchest → padchest` | In-domain | 11,111 | 0.8486 → 0.6662 | 0.8636 → 0.6636 | −0.2000 | 0.2433 → 0.1081 |
| `mimic_ch → padchest` | Cross-domain | 11,111 | 0.7536 → 0.5680 | 0.7738 → 0.5893 | −0.1845 | 0.1488 → 0.0646 |
| `nih → padchest` | Cross-domain | 11,111 | **0.7639** → 0.5778 | **0.7832** → 0.5929 | −0.1904 | **0.1788** → 0.0821 |
| `chexpert → cxr_rait` | Cross-domain | 3,248 | 0.6806 → 0.6718 | 0.6806 → 0.6718 | −0.0089 | 0.3226 → 0.3197 |
| `padchest → cxr_rait` | Cross-domain | 3,248 | **0.7000** → 0.6473 | **0.7000** → 0.6473 | −0.0527 | **0.3859** → 0.3346 |
| `mimic_ch → cxr_rait` | Cross-domain | 3,248 | 0.6459 → 0.5834 | 0.6459 → 0.5834 | −0.0624 | 0.2865 → 0.2403 |
| `nih → cxr_rait` | Cross-domain | 3,248 | 0.6613 → 0.6282 | 0.6613 → 0.6282 | −0.0331 | 0.3234 → 0.3129 |
| `chexpert → mimic_cxr` | Cross-domain | 1,419 | **0.7017** → 0.6933 | **0.7138** → 0.7030 | −0.0108 | 0.3185 → 0.3131 |
| `padchest → mimic_cxr` | Cross-domain | 1,419 | 0.6997 → 0.6866 | 0.7111 → 0.6975 | −0.0135 | **0.3361** → 0.3254 |
| `mimic_ch → mimic_cxr` | In-domain | 1,419 | 0.7012 → 0.6452 | 0.7122 → 0.6580 | −0.0542 | 0.3117 → 0.2782 |
| `nih → mimic_cxr` | Cross-domain | 1,419 | 0.6668 → 0.6360 | 0.6762 → 0.6461 | −0.0301 | 0.2996 → 0.2885 |
| `chexpert → xraydar` | Cross-domain | 979 | 0.6727 → **0.6921** | 0.6727 → **0.6921** | **+0.0193** | 0.5328 → **0.5461** |
| `padchest → xraydar` | Cross-domain | 979 | **0.7527** → 0.7345 | **0.7527** → 0.7345 | −0.0182 | 0.5845 → **0.5870** |
| `mimic_ch → xraydar` | Cross-domain | 979 | 0.6592 → 0.6396 | 0.6592 → 0.6396 | −0.0196 | 0.5142 → 0.4960 |
| `nih → xraydar` | Cross-domain | 979 | 0.6815 → 0.6653 | 0.6815 → 0.6653 | −0.0162 | 0.5499 → 0.5450 |
| `chexpert → nih` | Cross-domain | 25,596 | 0.6355 → 0.6153 | 0.6457 → 0.6248 | −0.0210 | 0.1389 → 0.1407 |
| `padchest → nih` | Cross-domain | 25,596 | **0.7064** → 0.6011 | **0.7126** → 0.5980 | −0.1147 | **0.1866** → 0.1251 |
| `mimic_ch → nih` | Cross-domain | 25,596 | 0.6262 → 0.5195 | 0.6346 → 0.5404 | −0.0942 | 0.1425 → 0.1043 |
| `nih → nih` | In-domain | 25,596 | 0.6960 → 0.5725 | 0.7025 → 0.5626 | −0.1399 | 0.1913 → 0.1163 |

---

## 3. Per-Pathology Breakdown (`No Adapt.` Baseline)

### 3.1 Per-Pathology AUROC

| Pathology | CheX → PadChest | PadChest → CheX | CheX → RAIT | PadChest → RAIT | CheX → MIMIC | PadChest → MIMIC | CheX → X-Raydar | PadChest → X-Raydar | MIMIC → X-Raydar | CheX → NIH | PadChest → NIH | MIMIC → NIH | CheX → CheX (ref) | PadChest → PadChest (ref) | MIMIC → MIMIC (ref) | NIH → NIH (ref) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Atelectasis | 0.7013 | 0.5966 | 0.6183 | 0.6528 | 0.6685 | 0.6808 | 0.4645 | 0.5217 | 0.4793 | 0.5659 | 0.6549 | 0.6018 | 0.5811 | 0.7687 | 0.6912 | 0.6620 |
| Cardiomegaly | 0.8018 | 0.7142 | 0.7952 | 0.8809 | 0.7512 | 0.7511 | 0.7332 | 0.8938 | 0.7391 | 0.6229 | 0.7959 | 0.6008 | 0.7264 | 0.8952 | 0.7177 | 0.8079 |
| Pleural Effusion | 0.8941 | 0.8080 | 0.7591 | 0.8027 | 0.8093 | 0.8014 | 0.7477 | 0.8146 | 0.7406 | 0.6731 | 0.7517 | 0.6972 | 0.8375 | 0.9405 | 0.8023 | 0.7466 |
| Consolidation | 0.7551 | 0.6559 | 0.5457 | 0.5482 | 0.6657 | 0.6576 | 0.7632 | 0.7502 | 0.6667 | 0.6744 | 0.6543 | 0.6242 | 0.6818 | 0.8685 | 0.6608 | 0.6634 |
| Pneumothorax | 0.5910 | 0.6215 | 0.6850 | 0.6154 | 0.6140 | 0.6074 | 0.6550 | 0.7831 | 0.6705 | 0.6413 | 0.6751 | 0.6067 | 0.7338 | 0.7703 | 0.6342 | 0.5999 |
| Edema | 0.9046 | 0.7184 | — | — | 0.7743 | 0.7679 | — | — | — | 0.6968 | 0.7440 | 0.6767 | 0.7429 | 0.9383 | 0.7668 | 0.7352 |

### 3.2 Target Cohort Positive Prevalence

| Pathology | PadChest (`N = 11,111`) | CheXpert (`N = 21,438`) | CXR-RAIT (`N = 3,248`) | MIMIC-CXR (`N = 1,419`) | X-Raydar (`N = 979`) | NIH (`N = 25,596`) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Atelectasis | 596 (5.36%) | 3,154 (14.71%) | 215 (6.62%) | 420 (29.60%) | 254 (25.94%) | 3,279 (12.81%) |
| Cardiomegaly | 1,008 (9.07%) | 4,889 (22.81%) | 567 (17.46%) | 246 (17.34%) | 632 (64.56%) | 1,069 (4.18%) |
| Pleural Effusion | 717 (6.45%) | 15,510 (72.35%) | 534 (16.44%) | 463 (32.63%) | 548 (55.98%) | 4,658 (18.20%) |
| Consolidation | 180 (1.62%) | 1,371 (6.40%) | 1,874 (57.70%) | 113 (7.96%) | 501 (51.17%) | 1,815 (7.09%) |
| Pneumothorax | 39 (0.35%) | 1,294 (6.04%) | 18 (0.55%) | 53 (3.74%) | 43 (4.39%) | 2,665 (10.41%) |
| Edema | 137 (1.23%) | 6,226 (29.04%) | — | 244 (17.20%) | — | 925 (3.61%) |

On PadChest (`N = 11,111`), both `CheXpert → PadChest` (`0.7919` vs. `0.7734`) and `PadChest → PadChest` (`0.8908` vs. `0.8559`) achieve higher 6-pathology macro AUROC on physician-curated labels (`MethodLabel=Physician`, `N = 2,578`) than on automated RNN labels (`MethodLabel=RNN_model`, `N = 8,533`).

---

## 4. Analysis and Discussion

### 4.1 Comparison with Published Literature (MICCAI 2026 CoWA)

Comparing our results against Table 1 of Jeong et al. (*"Leveraging Pathology Co-occurrence for Test-Time Adaptation in Chest X-Ray Diagnosis"*, MICCAI 2026, [Paper 4786](https://papers.miccai.org/miccai-2026/paper/4786_paper.pdf)):
- **Exact baseline parity on `CheXpert → NIH` (`test_list.txt`):** Our unadapted `CheXpert → NIH` AUROC (`0.5659 / 0.6229 / 0.6731 / 0.6744 / 0.6413 / 0.6968`, macro **0.6457**) matches the paper's `No Adapt.` baseline (`56.6 / 62.3 / 67.4 / 67.4 / 64.1 / 69.7`, macro **64.6**) to within `0.0009` per pathology, validating our image preprocessing and raw-logit scoring.
- **Unadapted PadChest surpasses adapted CheXpert on NIH:** Jeong et al. only evaluated CheXpert and NIH as sources. Our unadapted `PadChest → NIH` model (**0.7126** macro AUROC) outperforms both unadapted `CheXpert → NIH` (`64.6`) and CoWA-adapted `CheXpert → NIH` (`66.8`), as well as the in-domain NIH model (`0.7025`).
- **Cohort differences on `CheXpert → MIMIC-CXR`:** Our `CheXpert → MIMIC-CXR` baseline achieves **0.7138** macro AUROC vs. **66.7** in MICCAI 2026 because our cohort uses the 1,419 **frontal-only (`AP`/`PA`)** validation images from Kaggle `simhadrisadaram/mimic-cxr-dataset` with regex-parsed labels, whereas MICCAI 2026 used the official PhysioNet `MIMIC-CXR-JPG` unseen test split.

### 4.2 Why CoWA Degrades on Long Uncalibrated Streams

Our CoWA TTA sweep reveals a clear relationship between target stream length (`N / 64` batches) and adaptation stability when using a fixed learning rate (`lr = 1e-3`) without XRV's `op_threshs` calibration:
- **Short streams (`16–51` batches):** On `CheXpert → X-Raydar` (`N = 979`, 16 batches), CoWA improves macro AUROC (`0.6727 → 0.6921`, **+0.0193**) and macro AUPRC (`0.5328 → 0.5461`). On `PadChest → X-Raydar`, macro AUPRC ticks up slightly (`0.5845 → 0.5870`) while AUROC dips (`−0.0182`).
- **Long streams (`174–400` batches):** On PadChest (174 batches), CheXpert (335 batches), and NIH (400 batches), running `lr = 1e-3` continuously without per-pair learning-rate tuning (`[1e-4, 1e-2]` in Jeong et al., whose `configs/tta_ours_configs_final.json` is omitted from GitHub) or XRV's operating-point calibration (`op_threshs`) causes cumulative entropy-minimization drift (`−0.086` to `−0.200` macro AUROC). Tuning smaller learning rates (e.g. `1e-4`) or resetting/bounding BN updates is therefore essential when deploying CoWA on large target cohorts.

---

## 5. Reproducibility and Artifacts

All scripts run in `/mnt/data/miniconda/envs/crl-cxr` on TorchTPU:

```bash
# Unadapted per-cohort evaluations (compiled TorchTPU inference)
/mnt/data/miniconda/envs/crl-cxr/bin/python scripts/evaluate_torchxrayvision_chexpert_padchest.py --device tpu --compile-model --batch-size 64
/mnt/data/miniconda/envs/crl-cxr/bin/python scripts/evaluate_torchxrayvision_rait_5pathologies.py --device tpu --compile-model --batch-size 64
/mnt/data/miniconda/envs/crl-cxr/bin/python scripts/evaluate_torchxrayvision_mimic.py --device tpu --compile-model --batch-size 64
/mnt/data/miniconda/envs/crl-cxr/bin/python scripts/evaluate_torchxrayvision_xraydar.py --device tpu --compile-model --batch-size 64
/mnt/data/miniconda/envs/crl-cxr/bin/python scripts/evaluate_torchxrayvision_nih.py --device tpu --compile-model --batch-size 64 --num-workers 8

# Full 24-pair CoWA TTA vs. No Adapt. sweep
/mnt/data/miniconda/envs/crl-cxr/bin/python scripts/evaluate_cowa_cross_domain.py --device tpu --batch-size 64
```

Machine-readable outputs (`summary.json`, `comparison.csv`, and per-pair predictions) are stored under `chexpert/runs/`:
- `chexpert/runs/xrv_densenet121_chexpert_padchest/`
- `chexpert/runs/xrv_densenet121_cxr_rait_5pathologies/`
- `chexpert/runs/xrv_densenet121_mimic_cxr/`
- `chexpert/runs/xrv_densenet121_xraydar/`
- `chexpert/runs/xrv_densenet121_nih/`
- `chexpert/runs/cowa_densenet121_cross_domain/`
