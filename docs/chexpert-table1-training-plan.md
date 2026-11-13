# CheXpert Table 1 Replication Training Plan

## Objective and scope

This plan targets the paper's main classification result: the **full CheXpert AUROC column of Table 1** in [arXiv:2506.20582](https://arxiv.org/abs/2506.20582). It excludes MIMIC-CXR and cross-dataset evaluation.

| Axis | Table 1 scope |
|---|---|
| Binary tasks | No Finding vs Pleural Effusion; No Finding vs Cardiomegaly |
| Backbones | ResNet-18, DenseNet-121, EfficientNet-B0 |
| Conditions | Baseline, view-invariant, race-invariant, sex-invariant |
| Configurations | 2 tasks × 3 backbones × 4 conditions = 24 |
| Replicates | 10 seeds per configuration = 240 training runs |
| Reported seeds | Top 5 of 10 per configuration, selected by validation AUROC |

Section 3.2 is the table-specific source for selecting the top five seeds by validation AUROC. The implementation-details paragraph instead says to select the model with the “largest” validation BCE. This plan follows Section 3.2 for Table 1 and records the conflicting statement as a protocol discrepancy. The paper does not name the ten seed values, give an exact invariant group size beyond P ≥ 2, or specify a clear within-seed epoch checkpoint rule. Those choices below are explicit interpretations, not paper-specified facts.

A five-step ResNet-18 race-invariance smoke test has already passed on TPU using a 320-image subset. It is an infrastructure check only, not a paper result. The complete Table 1 run has not started.

## Paper settings and operational choices

| Setting | Plan |
|---|---|
| Input size | 224 × 224 |
| Optimizer | Adam |
| Feature-extractor learning rate | 0.001 |
| Classifier learning rate | 0.0001 |
| Epochs | 20 |
| Batch size | 32; document whether the implementation counts images or invariant groups |
| Invariant group size | P = 2, the minimum satisfying P ≥ 2; disclose this interpretation |
| Seeds | 42–51, an explicit reproducibility choice because the paper gives no seed values |
| Within-seed checkpoint | Maximum validation AUROC as an operational choice; disclose the paper's conflicting BCE sentence |
| Seed selection | Rank all 10 runs by validation AUROC; select the top 5 |
| Final metric | Test AUROC from continuous positive-class scores; report the selected-five mean and standard deviation |

Use ImageNet-pretrained ResNet-18, DenseNet-121, and EfficientNet-B0. For each task/backbone/seed, keep the split, optimizer, schedule, and evaluation fixed across conditions; change only baseline versus the specified protected-attribute invariance condition. Do not use test results to choose epochs, seeds, thresholds, or hyperparameters.

The prepared Python 3.13/TorchTPU environment is the execution target. The installed Lightning TPU/XLA path is unavailable in this environment, so use the working native TorchTPU eager execution path rather than silently falling back to CPU. Record the actual Python, PyTorch, TorchTPU, Lightning, accelerator, and driver versions.

## Data source, labels, and split verification

Use the owner-authorized CheXpert source with the existing `datafiles/chexpert/chexpert_split_2021_08_20.csv` split membership. Keep credentials out of the repository. Do not write to the source bucket or commit images, patient-level labels, demographic records, or credentials.

The current data audit found that the original `train.csv` embedded in the CheXpert archive, joined to the repository split by row index, reproduces the paper's Table 3 counts. The separately provided CheXbert and VisualCheXbert label CSVs do not reproduce those counts under that join; do not substitute them without new evidence. Preserve the source file identity, row count, join rule, and hashes in the private experiment record so the mapping is auditable.

Verify these counts before staging, separately for each binary task:

| Label | Train | Validation | Test |
|---|---:|---:|---:|
| No Finding | 6,514 | 1,086 | 3,316 |
| Pleural Effusion | 31,015 | 5,049 | 15,510 |
| Cardiomegaly | 9,353 | 1,548 | 4,889 |

Apply the cohort and label filters used by the project notebook, then confirm the counts remain aligned. Check duplicate images, patient/study identity, split leakage, missing labels, uncertain-label treatment, and target co-occurrence. Verify the race, sex, and view mappings from their source columns; do not infer the meaning of numeric codes. Record counts for each target-by-protected-group-by-split cell, including any empty cells that would make invariant sampling invalid.

## Storage-conscious image preparation

The source archives are too large to download wholesale. The current audit maps 71,096 unique images across both tasks and all splits, with 18,100 images shared between the task subsets. Their original JPEG payloads alone exceed the currently available local space. A conservative uncompressed pixel-data bound for 224 × 224 grayscale output is about 3.57 GB for this image union, before image-container overhead; actual space must be measured.

1. Recheck free space on the data volume before each staging operation. Keep the dataset, logs, caches, and checkpoints off the nearly full system volume.
2. Use byte-range access to the ZIP64 central directories to locate only required image entries. Verify entry names, sizes, CRCs, and source-object metadata; do not fetch the full ZIP archives.
3. Stream each selected JPEG, decode and resize it in memory using the project notebook's preprocessing, and stage only the required 224 × 224 grayscale output. Reuse the shared images across the two tasks. Convert grayscale to the models' expected three-channel input in the loader, not by duplicating raw source files.
4. Verify every selected path was mapped and decoded, the output manifest is complete, and image shape/intensity ranges are consistent. If selective reads fail, the subset does not fit, or required files are missing, stop with the evidence rather than training on an incomplete or guessed dataset.
5. Keep processed images and row-level manifests in a permission-restricted private staging location outside version control. Publish only aggregate counts and non-sensitive provenance.

One source JPEG has been range-read and decoded successfully in memory, but that alone does not prove every selected image is readable. Validate the complete required subset before the full run.

## Implementation and metric checks

The checked-in race-invariance entry point currently differs materially from the paper: its default epoch count is 10; its classifier learning rate is 0.001; checkpointing monitors minimum classification loss; and its metric helper computes ROC AUC from hard predicted classes. Before the full matrix:

- configure the two learning rates and 20-epoch schedule independently;
- add a deterministic runner for the 24-configuration × 10-seed matrix and save per-seed validation histories;
- select each seed's epoch by maximum validation AUROC as the disclosed operational choice, then rank seeds by that same validation metric;
- calculate AUROC from continuous positive-class scores and test it against an independent trusted implementation;
- check that invariant sampling uses reproducible, advancing random streams and that every required protected group has examples for both classes;
- compare the implemented invariant objective, group construction, and effective batch-size semantics with the paper; describe any unverified formula or sampling difference rather than claiming exact reproduction.

The current smoke run covers ResNet-18 with race-invariance only. Before the full matrix, smoke-test baseline loading and forward/backward/checkpoint/output paths for all three backbones, and exercise view, race, and sex group sampling on a bounded subset. Smoke metrics are diagnostic only.

## Full-run execution and selection

Run on the single TPU one training process at a time. Use a fixed configuration order and persist each run's config, seed, log, validation history, and checkpoint as it completes. A failed run is recorded and retried with the same configuration and seed; do not silently change the protocol to make a run pass.

For each of the 24 configurations:

1. Complete all 10 seeds using the same verified split and preprocessing.
2. Select each seed's checkpoint using validation AUROC only.
3. Rank the 10 seed runs by validation AUROC and select the top five, following Section 3.2.
4. Evaluate each selected checkpoint on the test split once.
5. Report the five test AUROCs, their mean and standard deviation, the ten validation AUROCs, and selected seed identities. Keep test results out of all selection decisions.

Do not add Table 2's separation metric, Figure 2's latent-density plot, MIMIC-CXR, or cross-dataset results to this Table 1 run.

## Deliverables and reporting

For each configuration and seed, preserve a machine-readable config, code revision, environment/device record, training log, checkpoint and hash, validation history, test predictions/logits, and metric outputs. Preserve the data-source/split manifest and preprocessing provenance privately. Produce a final table with one row per configuration, the ten validation AUROCs, five selected seeds, test mean ± standard deviation, verified split counts, and deviations from the paper.

Do not place images, patient-level data, credentials, or private machine paths in the repository. Label the outcome **adapted reproduction** if any material data, implementation, training, checkpoint, or metric difference remains; do not describe smoke metrics as paper results or claim exact reproduction without evidence for every protocol element.
