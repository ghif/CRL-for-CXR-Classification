# Cross-Domain Chest X-Ray Classification and Test-Time Adaptation with Pretrained DenseNet-121

## Abstract

We present a systematic cross-domain evaluation of four pretrained TorchXRayVision DenseNet-121 classifiers—trained on **CheXpert**, **PadChest**, **MIMIC-CXR**, and **NIH ChestX-ray14**—across **six target cohorts** (**CheXpert**, **PadChest**, **CXR-RAIT**, **MIMIC-CXR**, **X-Raydar**, and **NIH ChestX-ray14**, totaling **63,791** test radiographs) on **TorchTPU**. Across all **24 source–target pairs** (20 cross-domain and 4 in-domain reference baselines), we benchmark **unadapted zero-shot transfer (`No Adapt.`)** against **six source-free test-time adaptation (TTA) algorithms**: **AdaBN**, **TENT**, **CoTTA**, **EATA**, **RoTTA**, and **CoWA**.

Our experiments reveal four main findings:
1. **PadChest generalizes best across external domains:** Without adaptation, `densenet121-res224-pc` achieves the highest cross-domain macro AUROC on **NIH ChestX-ray14** (`0.7126`, exceeding even the in-domain NIH model at `0.7025`), **X-Raydar** (`0.7527`), and **CXR-RAIT** (`0.7000`).
2. **Exact alignment with published benchmarks:** On the official NIH `test_list.txt` split, our `CheXpert → NIH` unadapted (`0.6457`) and AdaBN (`0.6685`) baselines match Table 1 of Jeong et al. (MICCAI 2026: `64.6` and `67.0`) to within `0.0015`.
3. **Stability vs. collapse across TTA families:** Across the 20 cross-domain pairs, **No Adapt.** (`0.6941 ± 0.0445`), **CoTTA** (`0.6921 ± 0.0454`), **AdaBN** (`0.6914 ± 0.0460`), and **RoTTA** (`0.6904 ± 0.0464`) remain stable, with **AdaBN** winning the most individual pairs (**9 of 24**, e.g., `chexpert → nih` `+0.0227`, `nih → cxr_rait` `+0.0222`). In contrast, pure or weakly regularized entropy-minimization methods (**TENT** `0.6308`, **EATA** `0.6350`, **CoWA** `0.6266`) improve short-stream transfer (`chexpert → xraydar`, 16 batches: `+0.0178` to `+0.0201`) but collapse on long streams (`174–400` batches) at fixed `lr = 1e-3` without `op_threshs` calibration.
4. **Runtime–accuracy trade-off:** Offline TTA adds **~3× to ~29×** wall-clock overhead over compiled `No Adapt.` inference, with **EATA** being the slowest (~6× slower than TENT) due to its two-pass Fisher estimation.

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

We evaluate **6 shared thoracic pathologies**: `Atelectasis`, `Cardiomegaly`, `Pleural Effusion` (`Effusion` head), `Consolidation`, `Pneumothorax`, and `Edema`. On **CXR-RAIT** and **X-Raydar**, `Edema` is omitted because no direct `edema` label exists in their annotation schemas; we therefore report both **5-pathology** (common to all 6 targets) and **all-available-pathology** macro averages.

### 1.2 Target Cohorts and Preprocessing

| Target Cohort | N | Batches (`B=64`) | Label Source & Binarization | Image Format & Normalization |
| --- | ---: | ---: | --- | --- |
| **CheXpert** | 21,438 | 335 | `chexpert.sample.test.csv` joined with HF `danjacobellis/chexpert` Parquet shards; U-Zeroes (`code == 3` → `1`) | 224×224 `uint8` PNG, `normalize(maxval=255)` |
| **PadChest** | 11,111 | 174 | `padchest-stage-manifest.jsonl` (`split == "test"`, frontal `PA`/`AP`/`AP_horizontal`) + `PADCHEST_..._160K_01.02.19.csv` | 224×224 **`uint16`** PNG, `normalize(maxval=65535)` |
| **CXR-RAIT** | 3,248 | 51 | `labels_5pathologies.csv` (`prediction_nlp >= 0.5` from `all_years_rekonsiliasi_final.csv`); 5 pathologies | 224×224 `uint8` PNG, `normalize(maxval=255)` |
| **MIMIC-CXR** | 1,419 | 23 | Frontal `AP`/`PA` from Kaggle `simhadrisadaram/mimic-cxr-dataset` (`mimic_cxr_aug_validate.csv`), negation-aware regex over `text` | 512×512 RGB JPEG → grayscale, `normalize(255)`, `XRayResizer(224)` |
| **X-Raydar** | 979 | 16 | `annotations.jsonl` radiologist-adjudicated `consensus_labels`; 5 pathologies | Full-res `uint8` PNG → `normalize(255)`, `XRayResizer(224)` |
| **NIH ChestX-ray14** | 25,596 | 400 | Official `test_list.txt` + `Data_Entry_2017.csv` from Kaggle `nih-chest-xrays/data`, pipe-delimited `Finding Labels` | 1024×1024 `uint8` PNG → `normalize(255)`, `XRayResizer(224)` |

### 1.3 Hardware, Environment, and TTA Protocols

All runs executed on **TorchTPU** (`/mnt/data/miniconda/envs/crl-cxr`, Python 3.13, `torch-tpu==0.2.0`, `bfloat16` autocast, `batch_size = 64`, `seed = 42`). Because TorchTPU cannot trace `op_threshs` (`aten.index_put`), probabilities during adaptation are computed via `torch.sigmoid(raw_logits)`. All gradient-based methods adapt only BatchNorm affine parameters (`weight`, `bias`) using Adam (`lr = 1e-3`, `steps = 1`), with `install_tpu_batchnorm_stats_fix` attached so `train()`-mode BatchNorm EMA updates are preserved:

- **AdaBN** (Li et al., ICLR-W 2017): Resets BN running stats (`momentum = None`) and accumulates cumulative target-domain mean/variance in `train()` mode (`no_grad`).
- **TENT** (Wang et al., ICLR 2021): Minimizes unweighted binary entropy $H(\hat{y})$.
- **CoTTA** (Wang et al., CVPR 2022): Student entropy minimization + teacher EMA (`0.999`) + stochastic restoration (`0.01`).
- **EATA** (Niu et al., ICML 2022): Two-pass adaptation—Pass 0 estimates diagonal Fisher information; Pass 1 minimizes entropy on low-entropy samples (`ent / log 2 < 0.4`) with Fisher regularization (`alpha = 50.0`).
- **RoTTA** (Yuan et al., CVPR 2023): Memory bank (`size = 64`, sample `16`) + robust BN EMA (`nu = 0.001`) + teacher–student cross-entropy (`ema = 0.999`).
- **CoWA** (Jeong et al., MICCAI 2026): Accumulates target pathology co-occurrence matrix (`threshold = 0.5`) and weights per-sample entropy by Frobenius consistency (`tau = 0.1`, `min_weight = 0.01`).

---

## 2. Main Results: Unadapted vs. Six TTA Methods (All 24 Pairs)

Table 1 reports **Macro AUROC** across all available pathologies per target (6 pathologies on CheXpert, PadChest, MIMIC-CXR, NIH; 5 pathologies on CXR-RAIT, X-Raydar). **Bold** marks the best method per row.

| Source → Target | Regime | N | `No Adapt.` | `AdaBN` | `TENT` | `CoTTA` | `EATA` | `RoTTA` | `CoWA` |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `chexpert → chexpert` | In-domain | 21,438 | **0.7173** | 0.7106 | 0.6443 | 0.7131 | 0.5793 | 0.7160 | 0.6314 |
| `padchest → chexpert` | Cross-domain | 21,438 | 0.6858 | **0.6940** | 0.5721 | 0.6779 | 0.5798 | 0.6705 | 0.5641 |
| `mimic_ch → chexpert` | Cross-domain | 21,438 | **0.6665** | 0.6442 | 0.5483 | 0.6614 | 0.5413 | 0.6548 | 0.5586 |
| `nih → chexpert` | Cross-domain | 21,438 | 0.6503 | **0.6674** | 0.5518 | 0.6426 | 0.5636 | 0.6346 | 0.5533 |
| `chexpert → padchest` | Cross-domain | 11,111 | 0.7747 | 0.7732 | 0.7551 | **0.7754** | 0.7702 | 0.7746 | 0.6024 |
| `padchest → padchest` | In-domain | 11,111 | 0.8636 | **0.8644** | 0.6662 | 0.8612 | 0.6788 | 0.8632 | 0.6636 |
| `mimic_ch → padchest` | Cross-domain | 11,111 | 0.7738 | **0.7778** | 0.5660 | 0.7728 | 0.5756 | 0.7697 | 0.5893 |
| `nih → padchest` | Cross-domain | 11,111 | 0.7832 | **0.7840** | 0.5959 | 0.7793 | 0.6037 | 0.7820 | 0.5929 |
| `chexpert → cxr_rait` | Cross-domain | 3,248 | 0.6806 | 0.6644 | 0.6668 | **0.6807** | 0.6623 | 0.6803 | 0.6718 |
| `padchest → cxr_rait` | Cross-domain | 3,248 | 0.7000 | **0.7037** | 0.6477 | 0.6995 | 0.6484 | 0.6999 | 0.6473 |
| `mimic_ch → cxr_rait` | Cross-domain | 3,248 | **0.6459** | 0.6181 | 0.5946 | 0.6457 | 0.6123 | 0.6455 | 0.5834 |
| `nih → cxr_rait` | Cross-domain | 3,248 | 0.6613 | **0.6835** | 0.6283 | 0.6603 | 0.6316 | 0.6598 | 0.6282 |
| `chexpert → mimic_cxr` | Cross-domain | 1,419 | **0.7138** | 0.7104 | 0.7125 | **0.7138** | 0.7092 | **0.7138** | 0.7030 |
| `padchest → mimic_cxr` | Cross-domain | 1,419 | 0.7111 | 0.7045 | 0.6975 | 0.7111 | 0.6949 | **0.7112** | 0.6975 |
| `mimic_ch → mimic_cxr` | In-domain | 1,419 | **0.7122** | 0.6894 | 0.6581 | **0.7122** | 0.6709 | **0.7122** | 0.6580 |
| `nih → mimic_cxr` | Cross-domain | 1,419 | 0.6762 | **0.6798** | 0.6465 | 0.6759 | 0.6488 | 0.6763 | 0.6461 |
| `chexpert → xraydar` | Cross-domain | 979 | 0.6727 | 0.6853 | **0.6928** | 0.6728 | 0.6905 | 0.6727 | 0.6921 |
| `padchest → xraydar` | Cross-domain | 979 | 0.7527 | 0.7328 | 0.7349 | **0.7528** | 0.7310 | **0.7528** | 0.7345 |
| `mimic_ch → xraydar` | Cross-domain | 979 | **0.6592** | 0.6334 | 0.6437 | **0.6592** | 0.6453 | 0.6590 | 0.6396 |
| `nih → xraydar` | Cross-domain | 979 | 0.6815 | **0.6846** | 0.6653 | 0.6812 | 0.6625 | 0.6815 | 0.6653 |
| `chexpert → nih` | Cross-domain | 25,596 | 0.6457 | **0.6685** | 0.5610 | 0.6402 | 0.5911 | 0.6417 | 0.6248 |
| `padchest → nih` | Cross-domain | 25,596 | **0.7126** | 0.6914 | 0.5971 | 0.7095 | 0.6024 | 0.7036 | 0.5980 |
| `mimic_ch → nih` | Cross-domain | 25,596 | **0.6346** | 0.6277 | 0.5372 | 0.6291 | 0.5362 | 0.6247 | 0.5404 |
| `nih → nih` | In-domain | 25,596 | **0.7025** | 0.6987 | 0.5626 | 0.6899 | 0.5749 | 0.6842 | 0.5626 |

### Aggregate Summary Across the 20 Cross-Domain Pairs

| Method | Mean AUROC | Std | Min | Max | Pairs Won (of 24) |
| --- | ---: | ---: | ---: | ---: | ---: |
| `No Adapt.` | **0.6941** | 0.0445 | 0.6346 | 0.7832 | 9 |
| `CoTTA` | 0.6921 | 0.0454 | 0.6291 | 0.7793 | 4 |
| `AdaBN` | 0.6914 | 0.0460 | 0.6181 | **0.7840** | **10** |
| `RoTTA` | 0.6904 | 0.0464 | 0.6247 | 0.7820 | 4 |
| `EATA` | 0.6350 | 0.0618 | 0.5362 | 0.7702 | 0 |
| `TENT` | 0.6308 | 0.0642 | 0.5372 | 0.7551 | 1 |
| `CoWA` | 0.6266 | 0.0541 | 0.5404 | 0.7345 | 0 |

---

## 3. Runtime and Computational Complexity

### 3.1 Asymptotic Complexity

Let $N$ be the target cohort size, $B$ the batch size, $T = \lceil N/B \rceil$ the number of batches, $L$ the number of BatchNorm layers ($L = 121$ in DenseNet-121), $P_{\text{BN}}$ the trainable BN affine parameters, $C$ the number of pathologies, and $C_{\text{fwd}}, C_{\text{bwd}}$ the cost of one forward and backward pass per batch:

| Method | Passes over Target Data | Complexity per Cohort | Gradient Updates? |
| --- | --- | --- | :---: |
| `No Adapt.` | 1 (`eval`) | $O(T \cdot C_{\text{fwd}})$ | No |
| `AdaBN` | 2 (1 `train` no-grad + 1 `eval`) | $O(2 T \cdot C_{\text{fwd}} + T \cdot L)$ | No |
| `TENT` | 2 (1 `train` fwd/bwd + 1 `eval`) | $O(T(C_{\text{fwd}} + C_{\text{bwd}}) + T \cdot C_{\text{fwd}})$ | Yes |
| `CoTTA` | 2 (1 `train` fwd/bwd + 1 `eval`) | $O(T(C_{\text{fwd}} + C_{\text{bwd}} + P) + T \cdot C_{\text{fwd}})$ | Yes |
| `RoTTA` | 2 (1 memory-bank teacher+student + 1 `eval`) | $O(T(C_{\text{fwd}}^{\text{mem}} + C_{\text{bwd}}^{\text{mem}} + P + L) + T \cdot C_{\text{fwd}})$ | Yes |
| `CoWA` | 2 (dual `eval`+`train` in Pass 1 + 1 `eval`) | $O(T(2 C_{\text{fwd}} + C_{\text{bwd}} + B C^2) + T \cdot C_{\text{fwd}})$ | Yes |
| `EATA` | **3** (1 Fisher fwd/bwd + 1 adapt fwd/bwd + 1 `eval`) | $O(2 T(C_{\text{fwd}} + C_{\text{bwd}} + P_{\text{BN}}) + T \cdot C_{\text{fwd}})$ | Yes |

### 3.2 Observed Median Wall-Clock Adaptation Time (TorchTPU, `batch_size = 64`)

Table 2 reports the **median adaptation-only wall-clock time** (in seconds, excluding Pass 2 evaluation and initial XLA compilation outliers) across the 4 source models for each target cohort, alongside compiled `No Adapt.` inference time where measured:

| Target Cohort | N | `No Adapt.` *(compiled)* | `TENT` | `CoTTA` | `RoTTA` | `AdaBN` | `CoWA` | `EATA` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **X-Raydar** | 979 | — | 1.41s | 1.72s | 2.14s | 3.37s | 3.76s | 11.38s |
| **MIMIC-CXR** | 1,419 | 3.80s | 2.02s | 2.38s | 2.85s | 4.80s | 5.23s | 15.88s |
| **CXR-RAIT** | 3,248 | 1.14s | 4.62s | 5.45s | 6.57s | 10.56s | 11.93s | 33.32s |
| **PadChest** | 11,111 | 5.03s | 15.58s | 18.25s | 21.51s | 36.27s | 40.57s | 101.35s |
| **CheXpert** | 21,438 | 7.47s | 28.91s | 32.54s | 42.95s | 73.48s | 74.80s | 191.13s |
| **NIH** | 25,596 | — | 37.06s | 43.11s | 48.85s | 86.12s | 93.38s | 230.58s |

Runtime scales linearly with $N$. In eager TorchTPU, **TENT** is the fastest TTA method (~3–4× slower than compiled `No Adapt.`), **CoTTA** and **RoTTA** add ~15–45% overhead for teacher EMA / memory sampling, **AdaBN** and **CoWA** take ~2.4–2.6× TENT (Python BN-EMA pre-hook over 121 layers / dual forward passes), and **EATA** is ~6.2× slower than TENT due to its full-dataset Fisher estimation pass.

---

## 4. Per-Pathology Breakdown (`No Adapt.` Baseline) and Cohort Prevalence

### 4.1 Per-Pathology AUROC (`No Adapt.`)

| Pathology | CheX → PadChest | PadChest → CheX | CheX → RAIT | PadChest → RAIT | CheX → MIMIC | PadChest → MIMIC | CheX → X-Raydar | PadChest → X-Raydar | MIMIC → X-Raydar | CheX → NIH | PadChest → NIH | MIMIC → NIH | CheX → CheX (ref) | PadChest → PadChest (ref) | MIMIC → MIMIC (ref) | NIH → NIH (ref) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Atelectasis | 0.7013 | 0.5966 | 0.6183 | 0.6528 | 0.6685 | 0.6808 | 0.4645 | 0.5217 | 0.4793 | 0.5659 | 0.6549 | 0.6018 | 0.5811 | 0.7687 | 0.6912 | 0.6620 |
| Cardiomegaly | 0.8018 | 0.7142 | 0.7952 | 0.8809 | 0.7512 | 0.7511 | 0.7332 | 0.8938 | 0.7391 | 0.6229 | 0.7959 | 0.6008 | 0.7264 | 0.8952 | 0.7177 | 0.8079 |
| Pleural Effusion | 0.8941 | 0.8080 | 0.7591 | 0.8027 | 0.8093 | 0.8014 | 0.7477 | 0.8146 | 0.7406 | 0.6731 | 0.7517 | 0.6972 | 0.8375 | 0.9405 | 0.8023 | 0.7466 |
| Consolidation | 0.7551 | 0.6559 | 0.5457 | 0.5482 | 0.6657 | 0.6576 | 0.7632 | 0.7502 | 0.6667 | 0.6744 | 0.6543 | 0.6242 | 0.6818 | 0.8685 | 0.6608 | 0.6634 |
| Pneumothorax | 0.5910 | 0.6215 | 0.6850 | 0.6154 | 0.6140 | 0.6074 | 0.6550 | 0.7831 | 0.6705 | 0.6413 | 0.6751 | 0.6067 | 0.7338 | 0.7703 | 0.6342 | 0.5999 |
| Edema | 0.9046 | 0.7184 | — | — | 0.7743 | 0.7679 | — | — | — | 0.6968 | 0.7440 | 0.6767 | 0.7429 | 0.9383 | 0.7668 | 0.7352 |

### 4.2 Target Cohort Positive Prevalence

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

## 5. Discussion and Comparison with Literature

1. **Alignment with MICCAI 2026 (Jeong et al., [Paper 4786](https://papers.miccai.org/miccai-2026/paper/4786_paper.pdf)):**
   - On `CheXpert → NIH` (`test_list.txt`), our `No Adapt.` (`0.6457`) and `AdaBN` (`0.6685`) macro AUROCs closely match Jeong et al.'s Table 1 (`64.6` and `67.0`).
   - On `CheXpert → MIMIC-CXR`, our `No Adapt.` (`0.7138` vs. `66.7`) reflects our frontal-only Kaggle validation cohort (`N = 1,419`) vs. the official PhysioNet `MIMIC-CXR-JPG` test split.
2. **Why `AdaBN`, `CoTTA`, and `RoTTA` outperform unconstrained entropy minimization (`TENT`, `EATA`, `CoWA`):**
   - Gradient-free `AdaBN` recalibrates feature normalization without risking parameter collapse, improving 10 of 24 pairs (including `+0.0227` on `chexpert → nih` and `+0.0222` on `nih → cxr_rait`).
   - `CoTTA` and `RoTTA` preserve stability via slow teacher EMA (`0.999`) and stochastic restoration / memory replay.
   - `TENT`, `EATA`, and `CoWA` with fixed `lr = 1e-3` and uncalibrated `sigmoid(raw_logits)` improve short-stream adaptation (`chexpert → xraydar`, 16 batches: `+0.0178` to `+0.0201`), but accumulate drift over 174–400 batches on PadChest, CheXpert, and NIH. Per-pair learning-rate tuning (`[1e-4, 1e-2]`) and operating-point calibration are therefore critical for gradient-based TTA on large cohorts.

---

## 6. Reproducibility and Artifacts

All scripts run in `/mnt/data/miniconda/envs/crl-cxr` on TorchTPU:

```bash
# Unadapted per-cohort evaluations (compiled TorchTPU inference)
/mnt/data/miniconda/envs/crl-cxr/bin/python scripts/evaluate_torchxrayvision_chexpert_padchest.py --device tpu --compile-model --batch-size 64
/mnt/data/miniconda/envs/crl-cxr/bin/python scripts/evaluate_torchxrayvision_rait_5pathologies.py --device tpu --compile-model --batch-size 64
/mnt/data/miniconda/envs/crl-cxr/bin/python scripts/evaluate_torchxrayvision_mimic.py --device tpu --compile-model --batch-size 64
/mnt/data/miniconda/envs/crl-cxr/bin/python scripts/evaluate_torchxrayvision_xraydar.py --device tpu --compile-model --batch-size 64
/mnt/data/miniconda/envs/crl-cxr/bin/python scripts/evaluate_torchxrayvision_nih.py --device tpu --compile-model --batch-size 64 --num-workers 8

# 24-pair TTA sweeps (CoWA, AdaBN, and TENT/CoTTA/EATA/RoTTA)
/mnt/data/miniconda/envs/crl-cxr/bin/python scripts/evaluate_cowa_cross_domain.py --device tpu --batch-size 64
/mnt/data/miniconda/envs/crl-cxr/bin/python scripts/evaluate_adabn_cross_domain.py --device tpu --batch-size 64
/mnt/data/miniconda/envs/crl-cxr/bin/python scripts/evaluate_tta_baselines_cross_domain.py --device tpu --batch-size 64
```

Outputs (`summary.json`, `comparison.csv`) are stored under `chexpert/runs/`:
- `chexpert/runs/cowa_densenet121_cross_domain/`
- `chexpert/runs/adabn_densenet121_cross_domain/`
- `chexpert/runs/tta_baselines_densenet121_cross_domain/`
