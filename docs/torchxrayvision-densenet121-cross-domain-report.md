# TorchXRayVision DenseNet-121 Cross-Domain Evaluation: CheXpert, PadChest, CXR-RAIT, and MIMIC-CXR

## Executive Summary

We evaluated three pretrained TorchXRayVision DenseNet-121 checkpoints—trained on **CheXpert** (`densenet121-res224-chex`), **PadChest** (`densenet121-res224-pc`), and **MIMIC-CXR** (`densenet121-res224-mimic_ch`, used as an in-domain reference on MIMIC-CXR)—across all six cross-domain transfer settings and three in-domain reference baselines on **TorchTPU** (`bfloat16`, `torch.compile(backend="tpu", dynamic=False)`, batch size 64):

- **CheXpert ↔ PadChest:** `CheXpert → PadChest` (`N = 11,111`), `PadChest → CheXpert` (`N = 21,438`) — 6 pathologies
- **CheXpert / PadChest → CXR-RAIT:** `CheXpert → CXR-RAIT` (`N = 3,248`), `PadChest → CXR-RAIT` (`N = 3,248`) — 5 pathologies (`Edema` omitted; absent from `prediction_nlp`)
- **CheXpert / PadChest → MIMIC-CXR:** `CheXpert → MIMIC-CXR` (`N = 1,419`), `PadChest → MIMIC-CXR` (`N = 1,419`) — 6 pathologies
- **In-domain references:** `CheXpert → CheXpert`, `PadChest → PadChest`, `MIMIC-CXR → MIMIC-CXR`

Neither cross-domain model was fine-tuned or calibrated on any target domain.

### Unified Summary (5-Pathology Common Subset & 6-Pathology Where Available)

The **5-pathology** macro average covers `Atelectasis`, `Cardiomegaly`, `Pleural Effusion`, `Consolidation`, and `Pneumothorax` (present across all four datasets). The **6-pathology** macro average adds `Edema` (available on CheXpert, PadChest, and MIMIC-CXR).

| Source → Target | Regime | Target N | 5-Path Macro AUROC | 5-Path Macro AUPRC | 6-Path Macro AUROC | 6-Path Macro AUPRC | Throughput (img/s) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **CheXpert → PadChest** | Cross-domain | 11,111 | **0.7487** | 0.1527 | **0.7747** | 0.1387 | 843.65 |
| **PadChest → CheXpert** | Cross-domain | 21,438 | 0.6792 | 0.3439 | 0.6858 | 0.3673 | 2,871.27 |
| **CheXpert → CXR-RAIT** | Cross-domain | 3,248 | 0.6806 | 0.3226 | — | — | 537.35* |
| **PadChest → CXR-RAIT** | Cross-domain | 3,248 | **0.7000** | **0.3859** | — | — | 2,846.37 |
| **CheXpert → MIMIC-CXR** | Cross-domain | 1,419 | **0.7017** | 0.3080 | **0.7138** | 0.3185 | 171.83* |
| **PadChest → MIMIC-CXR** | Cross-domain | 1,419 | 0.6997 | **0.3224** | 0.7111 | **0.3360** | 373.93 |
| CheXpert → CheXpert | In-domain (ref) | 21,438 | 0.7121 | 0.3513 | 0.7173 | 0.3736 | 464.99* |
| PadChest → PadChest | In-domain (ref) | 11,111 | 0.8486 | 0.2642 | 0.8636 | 0.2433 | 2,208.46 |
| MIMIC-CXR → MIMIC-CXR | In-domain (ref) | 1,419 | 0.7012 | 0.3009 | 0.7122 | 0.3117 | 372.56 |

*\*Includes initial TorchTPU XLA graph compilation time (`46.10s` for CheXpert, `6.04s` for CXR-RAIT, `8.26s` for MIMIC-CXR).*

---

## Key Cross-Domain Findings

1. **Both CheXpert and PadChest match in-domain performance on MIMIC-CXR:**
   - On the 1,419-image MIMIC-CXR frontal validation cohort, `CheXpert → MIMIC-CXR` (`0.7138` 6-path macro AUROC), `PadChest → MIMIC-CXR` (`0.7111`), and `MIMIC-CXR → MIMIC-CXR` (`0.7122`) all land within `0.0028` macro AUROC of one another, with `PadChest → MIMIC-CXR` achieving the highest macro AUPRC (`0.3360`).
2. **PadChest generalizes better than CheXpert to the external Indonesian cohort (CXR-RAIT):**
   - `PadChest → CXR-RAIT` outperforms `CheXpert → CXR-RAIT` on 5-pathology macro AUROC (**0.7000** vs. **0.6806**) and macro AUPRC (**0.3859** vs. **0.3226**), driven by large gains on `Cardiomegaly` (**0.8809** vs. `0.7952`, +0.0856) and `Pleural Effusion` (**0.8027** vs. `0.7591`, +0.0437).
   - `PadChest → CXR-RAIT` `Cardiomegaly` AUROC (`0.8809`) also exceeds the previously reported CXR-RAIT ResNet-18 baseline (`0.8464` in [the earlier CXR-RAIT report](torchxrayvision-densenet121-cxr-rait-cross-domain-report.md)).
3. **CheXpert ↔ PadChest transfer is asymmetric relative to in-domain baselines:**
   - `PadChest → CheXpert` lands within **−0.0315** 6-pathology macro AUROC of `CheXpert → CheXpert` (`0.6858` vs. `0.7173`) and achieves nearly identical macro AUPRC (`0.3673` vs. `0.3736`).
   - `CheXpert → PadChest` achieves strong absolute AUROC (`0.7747` 6-pathology macro; `0.9046` on `Edema`, `0.8941` on `Pleural Effusion`), but sits **−0.0889** below `PadChest → PadChest` (`0.8636`), with the largest gap on rare `Pneumothorax` (`0.5910` vs. `0.7703`, 39 positives).
4. **Consistent pathology-level patterns and failure modes:**
   - **Pleural Effusion** and **Edema** transfer most reliably across all datasets (`0.7591`–`0.9046` AUROC).
   - **Consolidation on CXR-RAIT** is near chance for both models (`0.5457` CheXpert, `0.5482` PadChest), reflecting CXR-RAIT's unusually high NLP-derived consolidation prevalence (`57.70%` vs. `1.62%`–`7.96%` elsewhere).
   - **Target prevalence governs AUPRC:** Low-prevalence targets such as PadChest frontal test (`0.35%`–`9.07%`) yield lower AUPRC (`0.14`–`0.26`) despite higher AUROC (`0.77`–`0.86`), whereas higher-prevalence CheXpert (`6.04%`–`72.35%`) and MIMIC-CXR (`3.74%`–`32.63%`) yield higher AUPRC (`0.31`–`0.37`).

---

## Per-Pathology Breakdown Across All Settings

### AUROC by Pathology and Setting

| Pathology | CheX → PadChest | PadChest → CheX | CheX → CXR-RAIT | PadChest → CXR-RAIT | CheX → MIMIC | PadChest → MIMIC | CheX → CheX (ref) | PadChest → PadChest (ref) | MIMIC → MIMIC (ref) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Atelectasis | 0.7013 | 0.5966 | 0.6183 | 0.6528 | 0.6685 | 0.6808 | 0.5811 | 0.7687 | 0.6912 |
| Cardiomegaly | 0.8018 | 0.7142 | 0.7952 | 0.8809 | 0.7512 | 0.7511 | 0.7264 | 0.8952 | 0.7177 |
| Pleural Effusion | 0.8941 | 0.8080 | 0.7591 | 0.8027 | 0.8093 | 0.8014 | 0.8375 | 0.9405 | 0.8023 |
| Consolidation | 0.7551 | 0.6559 | 0.5457 | 0.5482 | 0.6657 | 0.6576 | 0.6818 | 0.8685 | 0.6608 |
| Pneumothorax | 0.5910 | 0.6215 | 0.6850 | 0.6154 | 0.6140 | 0.6074 | 0.7338 | 0.7703 | 0.6342 |
| Edema | 0.9046 | 0.7184 | — | — | 0.7743 | 0.7679 | 0.7429 | 0.9383 | 0.7668 |
| **5-Path Macro** | **0.7487** | **0.6792** | **0.6806** | **0.7000** | **0.7017** | **0.6997** | 0.7121 | 0.8486 | 0.7012 |
| **6-Path Macro** | **0.7747** | **0.6858** | — | — | **0.7138** | **0.7111** | 0.7173 | 0.8636 | 0.7122 |

### AUPRC (Average Precision) by Pathology and Setting

| Pathology | CheX → PadChest | PadChest → CheX | CheX → CXR-RAIT | PadChest → CXR-RAIT | CheX → MIMIC | PadChest → MIMIC | CheX → CheX (ref) | PadChest → PadChest (ref) | MIMIC → MIMIC (ref) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Atelectasis | 0.1241 | 0.1856 | 0.1014 | 0.1055 | 0.4162 | 0.4220 | 0.1707 | 0.1571 | 0.4147 |
| Cardiomegaly | 0.2245 | 0.4491 | 0.4480 | 0.6619 | 0.3142 | 0.3731 | 0.4312 | 0.4597 | 0.2726 |
| Pleural Effusion | 0.3645 | 0.9001 | 0.4229 | 0.5440 | 0.6230 | 0.6296 | 0.9052 | 0.5640 | 0.6045 |
| Consolidation | 0.0436 | 0.0992 | 0.6257 | 0.6097 | 0.1261 | 0.1302 | 0.1085 | 0.1247 | 0.1440 |
| Pneumothorax | 0.0069 | 0.0853 | 0.0149 | 0.0083 | 0.0607 | 0.0569 | 0.1408 | 0.0152 | 0.0690 |
| Edema | 0.0684 | 0.4845 | — | — | 0.3709 | 0.4044 | 0.4851 | 0.1391 | 0.3653 |
| **5-Path Macro** | **0.1527** | **0.3439** | **0.3226** | **0.3859** | **0.3080** | **0.3224** | 0.3513 | 0.2642 | 0.3009 |
| **6-Path Macro** | **0.1387** | **0.3673** | — | — | **0.3185** | **0.3360** | 0.3736 | 0.2433 | 0.3117 |

### Target Cohort Sizes & Positive Prevalence

| Pathology | PadChest (`N = 11,111`) | CheXpert (`N = 21,438`) | CXR-RAIT (`N = 3,248`) | MIMIC-CXR (`N = 1,419`) |
| --- | ---: | ---: | ---: | ---: |
| Atelectasis | 596 (5.36%) | 3,154 (14.71%) | 215 (6.62%) | 420 (29.60%) |
| Cardiomegaly | 1,008 (9.07%) | 4,889 (22.81%) | 567 (17.46%) | 246 (17.34%) |
| Pleural Effusion | 717 (6.45%) | 15,510 (72.35%) | 534 (16.44%) | 463 (32.63%) |
| Consolidation | 180 (1.62%) | 1,371 (6.40%) | 1,874 (57.70%) | 113 (7.96%) |
| Pneumothorax | 39 (0.35%) | 1,294 (6.04%) | 18 (0.55%) | 53 (3.74%) |
| Edema | 137 (1.23%) | 6,226 (29.04%) | — | 244 (17.20%) |

---

## PadChest Label-Method Subgroup Check (`Physician` vs. `RNN_model`)

On PadChest (`N = 11,111`), both models achieve higher 6-pathology macro AUROC on physician-curated labels (`N = 2,578`) than on automated RNN-extracted labels (`N = 8,533`):
- **CheXpert → PadChest:** `0.7919` (`Physician`) vs. `0.7734` (`RNN_model`)
- **PadChest → PadChest:** `0.8908` (`Physician`) vs. `0.8559` (`RNN_model`)

---

## Experimental Setup, Data, and Normalization

| Dataset | Cohort & Label Extraction | Image Format & XRV Normalization |
| --- | --- | --- |
| **CheXpert** | `N = 21,438` (`datafiles/chexpert/chexpert.sample.test.csv` joined with `/mnt/data/dataset/chexpert/hf/data/train-*.parquet`, U-Zeroes: HF code `3` → `1`) | 224×224 `uint8` PNG (`/mnt/data/dataset/chexpert/hf_preprocessed/`), `normalize(maxval=255)` |
| **PadChest** | `N = 11,111` (`split == "test"` in `padchest-stage-manifest.jsonl`, frontal `PA`/`AP`/`AP_horizontal`, mapped from `PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv`) | 224×224 **`uint16`** PNG (`/mnt/data/dataset/padchest/input-stage/images-224/`), `normalize(maxval=65535)` |
| **CXR-RAIT** | `N = 3,248` (`/mnt/data/dataset/cxr_rait/labels_5pathologies.csv`, extracted from `all_years_rekonsiliasi_final - all_years_rekonsiliasi_final.csv` with `prediction_nlp >= 0.5`; `Edema` omitted) | 224×224 `uint8` PNG (`/mnt/data/dataset/cxr_rait/preprocessed_224/`), `normalize(maxval=255)` |
| **MIMIC-CXR** | `N = 1,419` (`/mnt/data/dataset/mimic_cxr/validate_frontal_6pathologies.csv`, frontal `AP`/`PA` from Kaggle `simhadrisadaram/mimic-cxr-dataset` `mimic_cxr_aug_validate.csv`, negation-aware regex over `text`) | 512×512 RGB `uint8` JPEG (`/mnt/data/dataset/mimic_cxr/official_data_iccv_final/files/`), grayscale + `normalize(maxval=255)` + `XRayResizer(224, engine="skimage")` |

### Models, Environment, and Reproduction

- **Checkpoints (`torchxrayvision==1.5.7`):**
  - `densenet121-res224-chex` (SHA-256: `1a914ae802170187a4852e594708206c12468eeeefd66aa4e8e26141649d61a4`)
  - `densenet121-res224-pc` (SHA-256: `a9148ef62ae4e7a31a9bed8cef22811f39681cd61e4390398936efc6280af521`)
  - `densenet121-res224-mimic_ch` (SHA-256: `23b13a04459684ffa41247d068207a7657b4e1b4dec2b02431f5026bd75b1189`)
- **Environment:** `/mnt/data/miniconda/envs/crl-cxr` (`crl-cxr-torchtpu`, Python 3.13, `torch-tpu==0.2.0`), `model.op_threshs = None`, `model.apply_sigmoid = False` (raw logits).
- **Limitations:** In-domain reference cells may overlap with TorchXRayVision's training splits. `CXR-RAIT` labels are NLP-derived (`prediction_nlp >= 0.5`), and the Kaggle `MIMIC-CXR` validation labels are regex-parsed from report text rather than official CheXpert-labeler CSVs.

```bash
# 1. CheXpert <-> PadChest (6 pathologies)
/mnt/data/miniconda/envs/crl-cxr/bin/python scripts/evaluate_torchxrayvision_chexpert_padchest.py \
  --device tpu --compile-model --batch-size 64 --num-workers 4

# 2. CheXpert / PadChest -> CXR-RAIT (5 pathologies)
/mnt/data/miniconda/envs/crl-cxr/bin/python scripts/evaluate_torchxrayvision_rait_5pathologies.py \
  --device tpu --compile-model --batch-size 64 --num-workers 4

# 3. CheXpert / PadChest / MIMIC-CXR -> MIMIC-CXR (6 pathologies)
/mnt/data/miniconda/envs/crl-cxr/bin/python scripts/evaluate_torchxrayvision_mimic.py \
  --device tpu --compile-model --batch-size 64 --num-workers 4
```
