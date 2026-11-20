# TorchXRayVision DenseNet-121 Cross-Domain Evaluation: CheXpert, PadChest, CXR-RAIT, MIMIC-CXR, X-Raydar, and NIH ChestX-ray14

## Executive Summary

We evaluated four pretrained TorchXRayVision DenseNet-121 checkpoints—trained on **CheXpert** (`densenet121-res224-chex`), **PadChest** (`densenet121-res224-pc`), **MIMIC-CXR** (`densenet121-res224-mimic_ch`), and **NIH ChestX-ray14** (`densenet121-res224-nih`)—across **12 cross-domain transfer settings** and **4 in-domain reference baselines** on **TorchTPU** (`bfloat16`, `torch.compile(backend="tpu", dynamic=False)`, batch size 64):

- **CheXpert ↔ PadChest:** `CheXpert → PadChest` (`N = 11,111`), `PadChest → CheXpert` (`N = 21,438`) — 6 pathologies
- **CheXpert / PadChest → CXR-RAIT:** `CheXpert → CXR-RAIT`, `PadChest → CXR-RAIT` (`N = 3,248`) — 5 pathologies (`Edema` omitted; absent from `prediction_nlp`)
- **CheXpert / PadChest → MIMIC-CXR:** `CheXpert → MIMIC-CXR`, `PadChest → MIMIC-CXR` (`N = 1,419`) — 6 pathologies
- **CheXpert / PadChest / MIMIC-CXR → X-Raydar:** `CheXpert → X-Raydar`, `PadChest → X-Raydar`, `MIMIC-CXR → X-Raydar` (`N = 979`, radiologist-consensus labels) — 5 pathologies (`Edema` omitted; absent from `consensus_labels`)
- **CheXpert / PadChest / MIMIC-CXR → NIH ChestX-ray14:** `CheXpert → NIH`, `PadChest → NIH`, `MIMIC-CXR → NIH` (`N = 25,596`, official `test_list.txt`) — 6 pathologies
- **In-domain references:** `CheXpert → CheXpert`, `PadChest → PadChest`, `MIMIC-CXR → MIMIC-CXR`, `NIH → NIH`

Neither cross-domain model was fine-tuned or calibrated on any target domain.

### Unified Summary (5-Pathology Common Subset & 6-Pathology Where Available)

The **5-pathology** macro average covers `Atelectasis`, `Cardiomegaly`, `Pleural Effusion`, `Consolidation`, and `Pneumothorax` (present across all six datasets). The **6-pathology** macro average adds `Edema` (available on CheXpert, PadChest, MIMIC-CXR, and NIH ChestX-ray14).

| Source → Target | Regime | Target N | 5-Path Macro AUROC | 5-Path Macro AUPRC | 6-Path Macro AUROC | 6-Path Macro AUPRC | Throughput (img/s) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **CheXpert → PadChest** | Cross-domain | 11,111 | **0.7487** | 0.1527 | **0.7747** | 0.1387 | 843.65 |
| **PadChest → CheXpert** | Cross-domain | 21,438 | 0.6792 | 0.3439 | 0.6858 | 0.3673 | 2,871.27 |
| **CheXpert → CXR-RAIT** | Cross-domain | 3,248 | 0.6806 | 0.3226 | — | — | 537.35* |
| **PadChest → CXR-RAIT** | Cross-domain | 3,248 | **0.7000** | **0.3859** | — | — | 2,846.37 |
| **CheXpert → MIMIC-CXR** | Cross-domain | 1,419 | **0.7017** | 0.3080 | **0.7138** | 0.3185 | 171.83* |
| **PadChest → MIMIC-CXR** | Cross-domain | 1,419 | 0.6997 | **0.3224** | 0.7111 | **0.3360** | 373.93 |
| **CheXpert → X-Raydar** | Cross-domain | 979 | 0.6727 | 0.5328 | — | — | 14.28† |
| **PadChest → X-Raydar** | Cross-domain | 979 | **0.7527** | **0.5845** | — | — | 14.40† |
| **MIMIC-CXR → X-Raydar** | Cross-domain | 979 | 0.6592 | 0.5142 | — | — | 14.40† |
| **CheXpert → NIH** | Cross-domain | 25,596 | 0.6355 | 0.1535 | 0.6457 | 0.1389 | 198.83‡ |
| **PadChest → NIH** | Cross-domain | 25,596 | **0.7064** | **0.2067** | **0.7126** | **0.1866** | 205.13‡ |
| **MIMIC-CXR → NIH** | Cross-domain | 25,596 | 0.6262 | 0.1585 | 0.6346 | 0.1425 | 205.56‡ |
| CheXpert → CheXpert | In-domain (ref) | 21,438 | 0.7121 | 0.3513 | 0.7173 | 0.3736 | 464.99* |
| PadChest → PadChest | In-domain (ref) | 11,111 | 0.8486 | 0.2642 | 0.8636 | 0.2433 | 2,208.46 |
| MIMIC-CXR → MIMIC-CXR | In-domain (ref) | 1,419 | 0.7012 | 0.3009 | 0.7122 | 0.3117 | 372.56 |
| NIH → NIH | In-domain (ref) | 25,596 | 0.6960 | 0.2128 | 0.7025 | 0.1913 | 205.29‡ |

*\*Includes initial TorchTPU XLA graph compilation time (`46.10s` for CheXpert, `6.04s` for CXR-RAIT, `8.26s` for MIMIC-CXR).*
*†Host-bound by full-resolution (~3520×4280) PNG decoding and on-the-fly resizing to 224×224 across 4 DataLoader workers (~68s per 979-image pass).*
*‡Host-bound by 1024×1024 PNG decoding and on-the-fly resizing to 224×224 across 8 DataLoader workers (~125s per 25,596-image pass).*

---

## Key Cross-Domain Findings

1. **PadChest is the strongest cross-domain source across external cohorts (NIH, X-Raydar, and CXR-RAIT):**
   - On **NIH ChestX-ray14** (`N = 25,596` official test split), `PadChest → NIH` achieves **0.7126** 6-path macro AUROC (**0.1866** macro AUPRC), **outperforming even the in-domain NIH model** (`0.7025` macro AUROC) and beating `CheXpert → NIH` (`0.6457`, +0.0669) and `MIMIC-CXR → NIH` (`0.6346`, +0.0781).
   - On **X-Raydar** (`N = 979`, radiologist-consensus labels), `PadChest → X-Raydar` achieves **0.7527** 5-path macro AUROC (**0.5845** macro AUPRC), outperforming `CheXpert → X-Raydar` (`0.6727`, +0.0800) and `MIMIC-CXR → X-Raydar` (`0.6592`, +0.0934).
   - On **CXR-RAIT** (`N = 3,248`, NLP-derived labels), `PadChest → CXR-RAIT` leads `CheXpert → CXR-RAIT` (`0.7000` vs. `0.6806` macro AUROC; `0.3859` vs. `0.3226` macro AUPRC).
2. **Both CheXpert and PadChest match in-domain performance on MIMIC-CXR:**
   - On the 1,419-image MIMIC-CXR frontal validation cohort, `CheXpert → MIMIC-CXR` (`0.7138` 6-path macro AUROC), `PadChest → MIMIC-CXR` (`0.7111`), and `MIMIC-CXR → MIMIC-CXR` (`0.7122`) all land within `0.0028` macro AUROC of one another, with `PadChest → MIMIC-CXR` achieving the highest macro AUPRC (`0.3360`).
3. **CheXpert ↔ PadChest transfer is asymmetric relative to in-domain baselines:**
   - `PadChest → CheXpert` lands within **−0.0315** 6-path macro AUROC of `CheXpert → CheXpert` (`0.6858` vs. `0.7173`) and achieves nearly identical macro AUPRC (`0.3673` vs. `0.3736`).
   - `CheXpert → PadChest` achieves strong absolute AUROC (`0.7747` 6-path macro; `0.9046` on `Edema`, `0.8941` on `Pleural Effusion`), but sits **−0.0889** below `PadChest → PadChest` (`0.8636`), with the largest gap on rare `Pneumothorax` (`0.5910` vs. `0.7703`, 39 positives).
4. **Dataset-specific pathology behavior & label-definition sensitivity:**
   - **Pleural Effusion** and **Edema** transfer reliably across all cohorts (`0.6731`–`0.9046` AUROC).
   - **Atelectasis on X-Raydar** is near/below chance for all three models (`0.4645`–`0.5217` AUROC), whereas **Consolidation on X-Raydar** transfers well (`0.7502`–`0.7632` for PadChest/CheXpert) in contrast to **Consolidation on CXR-RAIT** (`0.5457`–`0.5482`).

---

## Per-Pathology Breakdown Across All Settings

### AUROC by Pathology and Setting

| Pathology | CheX → PadChest | PadChest → CheX | CheX → RAIT | PadChest → RAIT | CheX → MIMIC | PadChest → MIMIC | CheX → X-Raydar | PadChest → X-Raydar | MIMIC → X-Raydar | CheX → NIH | PadChest → NIH | MIMIC → NIH | CheX → CheX (ref) | PadChest → PadChest (ref) | MIMIC → MIMIC (ref) | NIH → NIH (ref) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Atelectasis | 0.7013 | 0.5966 | 0.6183 | 0.6528 | 0.6685 | 0.6808 | 0.4645 | 0.5217 | 0.4793 | 0.5659 | 0.6549 | 0.6018 | 0.5811 | 0.7687 | 0.6912 | 0.6620 |
| Cardiomegaly | 0.8018 | 0.7142 | 0.7952 | 0.8809 | 0.7512 | 0.7511 | 0.7332 | 0.8938 | 0.7391 | 0.6229 | 0.7959 | 0.6008 | 0.7264 | 0.8952 | 0.7177 | 0.8079 |
| Pleural Effusion | 0.8941 | 0.8080 | 0.7591 | 0.8027 | 0.8093 | 0.8014 | 0.7477 | 0.8146 | 0.7406 | 0.6731 | 0.7517 | 0.6972 | 0.8375 | 0.9405 | 0.8023 | 0.7466 |
| Consolidation | 0.7551 | 0.6559 | 0.5457 | 0.5482 | 0.6657 | 0.6576 | 0.7632 | 0.7502 | 0.6667 | 0.6744 | 0.6543 | 0.6242 | 0.6818 | 0.8685 | 0.6608 | 0.6634 |
| Pneumothorax | 0.5910 | 0.6215 | 0.6850 | 0.6154 | 0.6140 | 0.6074 | 0.6550 | 0.7831 | 0.6705 | 0.6413 | 0.6751 | 0.6067 | 0.7338 | 0.7703 | 0.6342 | 0.5999 |
| Edema | 0.9046 | 0.7184 | — | — | 0.7743 | 0.7679 | — | — | — | 0.6968 | 0.7440 | 0.6767 | 0.7429 | 0.9383 | 0.7668 | 0.7352 |
| **5-Path Macro** | **0.7487** | **0.6792** | **0.6806** | **0.7000** | **0.7017** | **0.6997** | **0.6727** | **0.7527** | **0.6592** | **0.6355** | **0.7064** | **0.6262** | 0.7121 | 0.8486 | 0.7012 | 0.6960 |
| **6-Path Macro** | **0.7747** | **0.6858** | — | — | **0.7138** | **0.7111** | — | — | — | **0.6457** | **0.7126** | **0.6346** | 0.7173 | 0.8636 | 0.7122 | 0.7025 |

### AUPRC (Average Precision) by Pathology and Setting

| Pathology | CheX → PadChest | PadChest → CheX | CheX → RAIT | PadChest → RAIT | CheX → MIMIC | PadChest → MIMIC | CheX → X-Raydar | PadChest → X-Raydar | MIMIC → X-Raydar | CheX → NIH | PadChest → NIH | MIMIC → NIH | CheX → CheX (ref) | PadChest → PadChest (ref) | MIMIC → MIMIC (ref) | NIH → NIH (ref) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Atelectasis | 0.1241 | 0.1856 | 0.1014 | 0.1055 | 0.4162 | 0.4220 | 0.2332 | 0.2708 | 0.2435 | 0.1420 | 0.1920 | 0.1703 | 0.1707 | 0.1571 | 0.4147 | 0.2006 |
| Cardiomegaly | 0.2245 | 0.4491 | 0.4480 | 0.6619 | 0.3142 | 0.3731 | 0.7884 | 0.9315 | 0.7875 | 0.0620 | 0.1771 | 0.0564 | 0.4312 | 0.4597 | 0.2726 | 0.2229 |
| Pleural Effusion | 0.3645 | 0.9001 | 0.4229 | 0.5440 | 0.6230 | 0.6296 | 0.7808 | 0.8577 | 0.7856 | 0.2661 | 0.3820 | 0.3234 | 0.9052 | 0.5640 | 0.6045 | 0.3947 |
| Consolidation | 0.0436 | 0.0992 | 0.6257 | 0.6097 | 0.1261 | 0.1302 | 0.7655 | 0.7225 | 0.6780 | 0.1165 | 0.1012 | 0.0984 | 0.1085 | 0.1247 | 0.1440 | 0.1074 |
| Pneumothorax | 0.0069 | 0.0853 | 0.0149 | 0.0083 | 0.0607 | 0.0569 | 0.0961 | 0.1398 | 0.0763 | 0.1811 | 0.1813 | 0.1440 | 0.1408 | 0.0152 | 0.0690 | 0.1387 |
| Edema | 0.0684 | 0.4845 | — | — | 0.3709 | 0.4044 | — | — | — | 0.0660 | 0.0862 | 0.0625 | 0.4851 | 0.1391 | 0.3653 | 0.0838 |
| **5-Path Macro** | **0.1527** | **0.3439** | **0.3226** | **0.3859** | **0.3080** | **0.3224** | **0.5328** | **0.5845** | **0.5142** | **0.1535** | **0.2067** | **0.1585** | 0.3513 | 0.2642 | 0.3009 | 0.2128 |
| **6-Path Macro** | **0.1387** | **0.3673** | — | — | **0.3185** | **0.3360** | — | — | — | **0.1389** | **0.1866** | **0.1425** | 0.3736 | 0.2433 | 0.3117 | 0.1913 |

### Target Cohort Sizes & Positive Prevalence

| Pathology | PadChest (`N = 11,111`) | CheXpert (`N = 21,438`) | CXR-RAIT (`N = 3,248`) | MIMIC-CXR (`N = 1,419`) | X-Raydar (`N = 979`) | NIH (`N = 25,596`) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Atelectasis | 596 (5.36%) | 3,154 (14.71%) | 215 (6.62%) | 420 (29.60%) | 254 (25.94%) | 3,279 (12.81%) |
| Cardiomegaly | 1,008 (9.07%) | 4,889 (22.81%) | 567 (17.46%) | 246 (17.34%) | 632 (64.56%) | 1,069 (4.18%) |
| Pleural Effusion | 717 (6.45%) | 15,510 (72.35%) | 534 (16.44%) | 463 (32.63%) | 548 (55.98%) | 4,658 (18.20%) |
| Consolidation | 180 (1.62%) | 1,371 (6.40%) | 1,874 (57.70%) | 113 (7.96%) | 501 (51.17%) | 1,815 (7.09%) |
| Pneumothorax | 39 (0.35%) | 1,294 (6.04%) | 18 (0.55%) | 53 (3.74%) | 43 (4.39%) | 2,665 (10.41%) |
| Edema | 137 (1.23%) | 6,226 (29.04%) | — | 244 (17.20%) | — | 925 (3.61%) |

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
| **X-Raydar** | `N = 979` (`/mnt/data/dataset/xraydar/annotations.jsonl`, radiologist-adjudicated `consensus_labels`; `Edema` omitted) | Full-res `uint8` PNG (`/mnt/data/dataset/xraydar/images/`), `normalize(maxval=255)` + `XRayResizer(224, engine="skimage")` |
| **NIH ChestX-ray14** | `N = 25,596` (official `test_list.txt` + `Data_Entry_2017.csv` from Kaggle `nih-chest-xrays/data`, pipe-delimited `Finding Labels`) | 1024×1024 `uint8` PNG (`/mnt/data/dataset/nih_cxr/test_images/`), `normalize(maxval=255)` + `XRayResizer(224, engine="skimage")` |

### Models, Environment, and Reproduction

- **Checkpoints (`torchxrayvision==1.5.7`):**
  - `densenet121-res224-chex` (SHA-256: `1a914ae802170187a4852e594708206c12468eeeefd66aa4e8e26141649d61a4`)
  - `densenet121-res224-pc` (SHA-256: `a9148ef62ae4e7a31a9bed8cef22811f39681cd61e4390398936efc6280af521`)
  - `densenet121-res224-mimic_ch` (SHA-256: `23b13a04459684ffa41247d068207a7657b4e1b4dec2b02431f5026bd75b1189`)
  - `densenet121-res224-nih` (SHA-256: `31a5475c6b1383a7edbfcaf96af9ce302eb9969cfe28b6b8b5859afc6f274fdf`)
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

# 4. CheXpert / PadChest / MIMIC-CXR -> X-Raydar (5 pathologies)
/mnt/data/miniconda/envs/crl-cxr/bin/python scripts/evaluate_torchxrayvision_xraydar.py \
  --device tpu --compile-model --batch-size 64 --num-workers 4

# 5. CheXpert / PadChest / MIMIC-CXR / NIH -> NIH ChestX-ray14 (6 pathologies)
/mnt/data/miniconda/envs/crl-cxr/bin/python scripts/evaluate_torchxrayvision_nih.py \
  --device tpu --compile-model --batch-size 64 --num-workers 8
```
