# TorchXRayVision DenseNet-121 Cross-Domain Evaluation: CheXpert, PadChest, and CXR-RAIT

## Executive Summary

We evaluated two pretrained TorchXRayVision DenseNet-121 checkpoints—trained on **CheXpert** (`densenet121-res224-chex`) and **PadChest** (`densenet121-res224-pc`)—across four cross-domain transfer settings on **TorchTPU** (`bfloat16`, `torch.compile(backend="tpu", dynamic=False)`, batch size 64):
1. **CheXpert → PadChest** (`N = 11,111` frontal test images; 6 pathologies)
2. **PadChest → CheXpert** (`N = 21,438` test images; 6 pathologies)
3. **CheXpert → CXR-RAIT** (`N = 3,248` images; 5 pathologies, `Edema` omitted)
4. **PadChest → CXR-RAIT** (`N = 3,248` images; 5 pathologies, `Edema` omitted)

For reference, we also include the two in-domain baselines (**CheXpert → CheXpert** and **PadChest → PadChest**) run with the exact same pipeline. Neither model was fine-tuned or calibrated on any target domain.

### Unified Summary (5-Pathology Common Subset & 6-Pathology Where Available)

The 5-pathology macro average covers `Atelectasis`, `Cardiomegaly`, `Pleural Effusion`, `Consolidation`, and `Pneumothorax` (present in all three datasets). The 6-pathology macro average adds `Edema` (available on CheXpert and PadChest, omitted on CXR-RAIT because `edema` is absent from `prediction_nlp`).

| Source → Target | Regime | Target N | 5-Path Macro AUROC | 5-Path Macro AUPRC | 6-Path Macro AUROC | 6-Path Macro AUPRC | Throughput (img/s) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **CheXpert → PadChest** | Cross-domain | 11,111 | **0.7487** | 0.1527 | **0.7747** | 0.1387 | 843.65 |
| **PadChest → CheXpert** | Cross-domain | 21,438 | 0.6792 | 0.3439 | 0.6858 | 0.3673 | 2,871.27 |
| **CheXpert → CXR-RAIT** | Cross-domain | 3,248 | 0.6806 | 0.3226 | — | — | 537.35* |
| **PadChest → CXR-RAIT** | Cross-domain | 3,248 | **0.7000** | **0.3859** | — | — | 2,846.37 |
| CheXpert → CheXpert | In-domain (ref) | 21,438 | 0.7121 | 0.3513 | 0.7173 | 0.3736 | 464.99* |
| PadChest → PadChest | In-domain (ref) | 11,111 | 0.8486 | 0.2642 | 0.8636 | 0.2433 | 2,208.46 |

*\*Includes initial TorchTPU XLA graph compilation time (`46.10s` for CheXpert, `6.04s` for CXR-RAIT).*

---

## Key Cross-Domain Findings

1. **PadChest generalizes better than CheXpert to the external Indonesian cohort (CXR-RAIT):**
   - `PadChest → CXR-RAIT` outperforms `CheXpert → CXR-RAIT` on 5-pathology macro AUROC (**0.7000** vs. **0.6806**) and macro AUPRC (**0.3859** vs. **0.3226**), driven by large gains on `Cardiomegaly` (**0.8809** vs. `0.7952`, +0.0856) and `Pleural Effusion` (**0.8027** vs. `0.7591`, +0.0437).
   - Notably, `PadChest → CXR-RAIT` `Cardiomegaly` AUROC (`0.8809`) even exceeds the previously reported CXR-RAIT ResNet-18 baseline (`0.8464` in [the earlier CXR-RAIT report](torchxrayvision-densenet121-cxr-rait-cross-domain-report.md)).
2. **CheXpert ↔ PadChest transfer is asymmetric relative to in-domain baselines:**
   - `PadChest → CheXpert` lands within **−0.0315** 6-pathology macro AUROC of `CheXpert → CheXpert` (`0.6858` vs. `0.7173`) and achieves nearly identical macro AUPRC (`0.3673` vs. `0.3736`).
   - `CheXpert → PadChest` achieves strong absolute AUROC (`0.7747` 6-pathology macro), especially on `Edema` (`0.9046`), `Pleural Effusion` (`0.8941`), and `Cardiomegaly` (`0.8018`), but sits **−0.0889** below `PadChest → PadChest` (`0.8636`), with the largest gap on rare `Pneumothorax` (`0.5910` vs. `0.7703`, 39 positives).
3. **Pathology-level failure modes across domains:**
   - **Consolidation on CXR-RAIT** is near chance for both models (`0.5457` CheXpert, `0.5482` PadChest), likely due to CXR-RAIT's unusually high NLP-derived consolidation prevalence (`57.70%` vs. `6.40%` on CheXpert and `1.62%` on PadChest).
   - **Pneumothorax** has very low prevalence in both PadChest frontal test (`0.35%`, 39/11,111) and CXR-RAIT (`0.55%`, 18/3,248), leading to low AUPRC (`0.0069`–`0.0149`) and higher AUROC variance.
4. **Target prevalence strongly governs AUPRC:**
   - CheXpert's test split has high positive prevalence (e.g., `Pleural Effusion` `72.35%`, `Edema` `29.04%`), yielding higher AUPRC (`0.34`–`0.37`) even when AUROC is lower (`0.68`–`0.72`). PadChest's frontal test split has much lower prevalence (`0.35%`–`9.07%`), resulting in lower AUPRC (`0.14`–`0.26`) despite higher AUROC (`0.77`–`0.86`).

---

## Per-Pathology Breakdown Across All Settings

### AUROC by Pathology and Setting

| Pathology | CheXpert → PadChest | PadChest → CheXpert | CheXpert → CXR-RAIT | PadChest → CXR-RAIT | CheXpert → CheXpert (ref) | PadChest → PadChest (ref) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Atelectasis | 0.7013 | 0.5966 | 0.6183 | 0.6528 | 0.5811 | 0.7687 |
| Cardiomegaly | 0.8018 | 0.7142 | 0.7952 | 0.8809 | 0.7264 | 0.8952 |
| Pleural Effusion | 0.8941 | 0.8080 | 0.7591 | 0.8027 | 0.8375 | 0.9405 |
| Consolidation | 0.7551 | 0.6559 | 0.5457 | 0.5482 | 0.6818 | 0.8685 |
| Pneumothorax | 0.5910 | 0.6215 | 0.6850 | 0.6154 | 0.7338 | 0.7703 |
| Edema | 0.9046 | 0.7184 | — | — | 0.7429 | 0.9383 |
| **5-Path Macro AUROC** | **0.7487** | **0.6792** | **0.6806** | **0.7000** | 0.7121 | 0.8486 |
| **6-Path Macro AUROC** | **0.7747** | **0.6858** | — | — | 0.7173 | 0.8636 |

### AUPRC (Average Precision) & Target Prevalence

| Pathology | PadChest Prev. (N=11,111) | CheXpert → PadChest | PadChest → PadChest (ref) | CheXpert Prev. (N=21,438) | PadChest → CheXpert | CheXpert → CheXpert (ref) | CXR-RAIT Prev. (N=3,248) | CheXpert → CXR-RAIT | PadChest → CXR-RAIT |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Atelectasis | 5.36% | 0.1241 | 0.1571 | 14.71% | 0.1856 | 0.1707 | 6.62% | 0.1014 | 0.1055 |
| Cardiomegaly | 9.07% | 0.2245 | 0.4597 | 22.81% | 0.4491 | 0.4312 | 17.46% | 0.4480 | 0.6619 |
| Pleural Effusion | 6.45% | 0.3645 | 0.5640 | 72.35% | 0.9001 | 0.9052 | 16.44% | 0.4229 | 0.5440 |
| Consolidation | 1.62% | 0.0436 | 0.1247 | 6.40% | 0.0992 | 0.1085 | 57.70% | 0.6257 | 0.6097 |
| Pneumothorax | 0.35% | 0.0069 | 0.0152 | 6.04% | 0.0853 | 0.1408 | 0.55% | 0.0149 | 0.0083 |
| Edema | 1.23% | 0.0684 | 0.1391 | 29.04% | 0.4845 | 0.4851 | — | — | — |
| **5-Path Macro AUPRC** | — | **0.1527** | 0.2642 | — | **0.3439** | 0.3513 | — | **0.3226** | **0.3859** |
| **6-Path Macro AUPRC** | — | **0.1387** | 0.2433 | — | **0.3673** | 0.3736 | — | — | — |

---

## PadChest Label-Method Subgroup Check (`Physician` vs. `RNN_model`)

On PadChest (`N = 11,111`), both models achieve higher 6-pathology macro AUROC on physician-curated labels (`N = 2,578`) than on automated RNN-extracted labels (`N = 8,533`):
- **CheXpert → PadChest:** `0.7919` (`Physician`) vs. `0.7734` (`RNN_model`)
- **PadChest → PadChest:** `0.8908` (`Physician`) vs. `0.8559` (`RNN_model`)

---

## Data, Normalization, and Protocol

| Dataset | Test Cohort & Labels | Image Format & XRV Normalization |
| --- | --- | --- |
| **CheXpert** | `N = 21,438` (`datafiles/chexpert/chexpert.sample.test.csv` joined with `/mnt/data/dataset/chexpert/hf/data/train-*.parquet`, U-Zeroes binarization: HF code `3` → `1`) | 224×224 `uint8` PNG (`/mnt/data/dataset/chexpert/hf_preprocessed/`), `normalize(maxval=255)` |
| **PadChest** | `N = 11,111` (`split == "test"` in `padchest-stage-manifest.jsonl`, frontal projections `PA`, `AP`, `AP_horizontal`, mapped from `PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv`) | 224×224 **`uint16`** PNG (`/mnt/data/dataset/padchest/input-stage/images-224/`), `normalize(maxval=65535)` |
| **CXR-RAIT** | `N = 3,248` (`/mnt/data/dataset/cxr_rait/labels_5pathologies.csv`, extracted from `all_years_rekonsiliasi_final - all_years_rekonsiliasi_final.csv` with `prediction_nlp >= 0.5`; `Edema` omitted) | 224×224 `uint8` PNG (`/mnt/data/dataset/cxr_rait/preprocessed_224/`), `normalize(maxval=255)` |

### Models, Environment, and Reproduction
- **Checkpoints (`torchxrayvision==1.5.7`):**
  - `densenet121-res224-chex` (SHA-256: `1a914ae802170187a4852e594708206c12468eeeefd66aa4e8e26141649d61a4`)
  - `densenet121-res224-pc` (SHA-256: `a9148ef62ae4e7a31a9bed8cef22811f39681cd61e4390398936efc6280af521`)
- **Environment:** `/mnt/data/miniconda/envs/crl-cxr` (`crl-cxr-torchtpu`, Python 3.13, `torch-tpu==0.2.0`), `model.op_threshs = None`, `model.apply_sigmoid = False` (raw logits).
- **Limitations:** In-domain reference cells (`CheXpert → CheXpert`, `PadChest → PadChest`) may overlap with TorchXRayVision's training splits. `CXR-RAIT` labels are NLP-derived (`prediction_nlp >= 0.5`) rather than independently adjudicated clinical ground truth.

```bash
# CheXpert <-> PadChest (6 pathologies)
/mnt/data/miniconda/envs/crl-cxr/bin/python scripts/evaluate_torchxrayvision_chexpert_padchest.py \
  --device tpu --compile-model --batch-size 64 --num-workers 4

# CheXpert / PadChest -> CXR-RAIT (5 pathologies)
/mnt/data/miniconda/envs/crl-cxr/bin/python scripts/evaluate_torchxrayvision_rait_5pathologies.py \
  --device tpu --compile-model --batch-size 64 --num-workers 4
```
