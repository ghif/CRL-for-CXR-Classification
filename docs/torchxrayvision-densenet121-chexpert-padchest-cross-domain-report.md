# TorchXRayVision DenseNet-121 Cross-Domain Evaluation: CheXpert ↔ PadChest (6 Pathologies)

## Summary

We evaluated two pretrained TorchXRayVision DenseNet-121 checkpoints—trained on **CheXpert** (`densenet121-res224-chex`) and **PadChest** (`densenet121-res224-pc`)—across all four source → target combinations on **TorchTPU** (`bfloat16`, `torch.compile(backend="tpu", dynamic=False)`, batch size 64) for **6 shared pathologies**: `Atelectasis`, `Cardiomegaly`, `Pleural Effusion`, `Consolidation`, `Pneumothorax`, and `Edema`. Neither model was fine-tuned or calibrated on the target domain.

| Source → Target | Regime | Weights | Target Examples | Macro AUROC | Macro AUPRC | Throughput (img/s) |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| **CheXpert → PadChest** | Cross-domain | `densenet121-res224-chex` | 11,111 | **0.7747** | 0.1387 | 843.65 |
| **PadChest → CheXpert** | Cross-domain | `densenet121-res224-pc` | 21,438 | **0.6858** | 0.3673 | 2,871.27 |
| CheXpert → CheXpert | In-domain (ref) | `densenet121-res224-chex` | 21,438 | 0.7173 | 0.3736 | 464.99* |
| PadChest → PadChest | In-domain (ref) | `densenet121-res224-pc` | 11,111 | 0.8636 | 0.2433 | 2,208.46 |

*\*Includes initial TorchTPU XLA graph compilation time (~46.1s total for 21,438 images).*

Key takeaways:
1. **PadChest → CheXpert** transfers within **−0.0315 macro AUROC** of the CheXpert-trained model (`0.6858` vs. `0.7173`), with nearly identical macro AUPRC (`0.3673` vs. `0.3736`).
2. **CheXpert → PadChest** achieves **0.7747 macro AUROC** (strong transfer on `Edema` `0.9046`, `Pleural Effusion` `0.8941`, and `Cardiomegaly` `0.8018`), trailing the in-domain PadChest model (`0.8636`) by **−0.0889 macro AUROC**, with the largest drop on low-prevalence `Pneumothorax` (`0.5910` vs. `0.7703`, 39 positives).
3. **Target prevalence drives AUPRC:** CheXpert's test cohort has much higher pathology prevalence (e.g., `Pleural Effusion` 72.35%, `Edema` 29.04%) than PadChest's frontal test cohort (`Pleural Effusion` 6.45%, `Edema` 1.23%, `Pneumothorax` 0.35%), explaining why AUPRC is higher on CheXpert even when AUROC is higher on PadChest.

---

## Per-Pathology Results

### Target: PadChest Frontal Test (`N = 11,111`)

| Pathology | Positives | Prevalence | CheXpert → PadChest AUROC | CheXpert → PadChest AUPRC | PadChest → PadChest AUROC | PadChest → PadChest AUPRC | Δ AUROC (Cross − In) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Atelectasis | 596 | 5.36% | 0.7013 | 0.1241 | 0.7687 | 0.1571 | −0.0674 |
| Cardiomegaly | 1,008 | 9.07% | 0.8018 | 0.2245 | 0.8952 | 0.4597 | −0.0933 |
| Pleural Effusion | 717 | 6.45% | 0.8941 | 0.3645 | 0.9405 | 0.5640 | −0.0464 |
| Consolidation | 180 | 1.62% | 0.7551 | 0.0436 | 0.8685 | 0.1247 | −0.1133 |
| Pneumothorax | 39 | 0.35% | 0.5910 | 0.0069 | 0.7703 | 0.0152 | −0.1794 |
| Edema | 137 | 1.23% | 0.9046 | 0.0684 | 0.9383 | 0.1391 | −0.0337 |
| **Macro Average** | — | — | **0.7747** | **0.1387** | **0.8636** | **0.2433** | **−0.0889** |

### Target: CheXpert Test (`N = 21,438`)

| Pathology | Positives | Prevalence | PadChest → CheXpert AUROC | PadChest → CheXpert AUPRC | CheXpert → CheXpert AUROC | CheXpert → CheXpert AUPRC | Δ AUROC (Cross − In) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Atelectasis | 3,154 | 14.71% | 0.5966 | 0.1856 | 0.5811 | 0.1707 | +0.0154 |
| Cardiomegaly | 4,889 | 22.81% | 0.7142 | 0.4491 | 0.7264 | 0.4312 | −0.0122 |
| Pleural Effusion | 15,510 | 72.35% | 0.8080 | 0.9001 | 0.8375 | 0.9052 | −0.0295 |
| Consolidation | 1,371 | 6.40% | 0.6559 | 0.0992 | 0.6818 | 0.1085 | −0.0259 |
| Pneumothorax | 1,294 | 6.04% | 0.6215 | 0.0853 | 0.7338 | 0.1408 | −0.1123 |
| Edema | 6,226 | 29.04% | 0.7184 | 0.4845 | 0.7429 | 0.4851 | −0.0245 |
| **Macro Average** | — | — | **0.6858** | **0.3673** | **0.7173** | **0.3736** | **−0.0315** |

---

## PadChest Label-Method Subgroup Analysis (`Physician` vs. `RNN_model`)

PadChest metadata distinguishes physician-curated labels (`MethodLabel=Physician`, `N = 2,578` frontal test images) from automated RNN-extracted labels (`MethodLabel=RNN_model`, `N = 8,533` frontal test images). Both models achieve higher macro AUROC on the physician-labeled subset:

| Source → Target | Subgroup | Examples | Macro AUROC | Macro AUPRC |
| --- | --- | ---: | ---: | ---: |
| CheXpert → PadChest | `MethodLabel=Physician` | 2,578 | **0.7919** | 0.1632 |
| CheXpert → PadChest | `MethodLabel=RNN_model` | 8,533 | 0.7734 | 0.1362 |
| PadChest → PadChest | `MethodLabel=Physician` | 2,578 | **0.8908** | 0.2999 |
| PadChest → PadChest | `MethodLabel=RNN_model` | 8,533 | 0.8559 | 0.2338 |

---

## Data, Labels, and Preprocessing

1. **CheXpert Test Cohort (`N = 21,438`):**
   - Image list from `datafiles/chexpert/chexpert.sample.test.csv`, joined by normalized path against the 23 Hugging Face `danjacobellis/chexpert` Parquet shards in `/mnt/data/dataset/chexpert/hf/data/train-*.parquet` to obtain all 6 pathology labels.
   - HF integer codes (`3` = positive `1.0`, `2` = negative `0.0`, `1` = uncertain `-1.0`, `0` = blank `NaN`) were binarized using U-Zeroes (`code == 3` → `1`, else `0`).
   - Preprocessed 224×224 `uint8` PNGs from `/mnt/data/dataset/chexpert/hf_preprocessed/preproc_224x224/` were normalized with `xrv.datasets.normalize(img, maxval=255)` to `[-1024, 1024]`.
2. **PadChest Test Cohort (`N = 11,111`):**
   - Staged `split == "test"` items from `/mnt/data/dataset/padchest/input-stage/images-224/padchest-stage-manifest.jsonl` joined with `/mnt/data/dataset/padchest/metadata/PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv`, filtered to frontal projections (`PA`, `AP`, `AP_horizontal`).
   - PadChest `Labels` were mapped to the 6 tasks: `Atelectasis` (`atelectasis` and `*atelectasis` subtypes), `Cardiomegaly` (`cardiomegaly`), `Pleural Effusion` (`pleural effusion`, `loculated pleural effusion`, `loculated fissural effusion`), `Consolidation` (`consolidation`), `Pneumothorax` (`pneumothorax`, `hydropneumothorax`), and `Edema` (`pulmonary edema`).
   - Preprocessed 224×224 **`uint16`** PNGs from `/mnt/data/dataset/padchest/input-stage/images-224/gcloud-bulk/` were normalized with `xrv.datasets.normalize(img, maxval=65535)` to `[-1024, 1024]`.

---

## Models, Hardware, and Reproduction

| Model | TorchXRayVision Weights | Checkpoint SHA-256 |
| --- | --- | --- |
| CheXpert DenseNet-121 | `densenet121-res224-chex` | `1a914ae802170187a4852e594708206c12468eeeefd66aa4e8e26141649d61a4` |
| PadChest DenseNet-121 | `densenet121-res224-pc` | `a9148ef62ae4e7a31a9bed8cef22811f39681cd61e4390398936efc6280af521` |

- **Environment:** `/mnt/data/miniconda/envs/crl-cxr` (`crl-cxr-torchtpu`, Python 3.13, `torchxrayvision==1.5.7`, `torch-tpu==0.2.0`).
- **TPU Execution:** `torch.compile(model, backend="tpu", dynamic=False)`, `torch.autocast("tpu", dtype=torch.bfloat16)`, fixed `batch_size=64` with last-batch padding/trimming, `model.op_threshs = None`, `model.apply_sigmoid = False` (raw logits).
- **Limitations:** In-domain cells (`chexpert → chexpert`, `padchest → padchest`) are reference baselines only because potential overlap between TorchXRayVision's training splits and these test cohorts was not checked.

### Reproduction & Artifacts

- Evaluator: [`scripts/evaluate_torchxrayvision_chexpert_padchest.py`](../scripts/evaluate_torchxrayvision_chexpert_padchest.py)
- Outputs: `chexpert/runs/xrv_densenet121_chexpert_padchest/{summary.json,comparison.csv,predictions_*.csv}`

```bash
/mnt/data/miniconda/envs/crl-cxr/bin/python scripts/evaluate_torchxrayvision_chexpert_padchest.py \
  --device tpu \
  --compile-model \
  --batch-size 64 \
  --num-workers 4
```
