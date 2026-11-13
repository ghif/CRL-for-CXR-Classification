#!/usr/bin/env python3
"""Evaluate a TorchXRayVision DenseNet-121 on the CheXpert test split."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

import torchxrayvision as xrv

from prediction.datasets.cxr import CXRDataset


REPO = Path(__file__).resolve().parents[1]
RUN_ROOT = REPO / "chexpert/runs/resnet18_baseline_tpu_ce"
CSV_ROOT = REPO / "datafiles/chexpert"
IMAGE_ROOT = "chexpert/hf_preprocessed/"
WEIGHTS = os.environ.get("CXR_XRV_WEIGHTS", "densenet121-res224-mimic_ch")
WEIGHT_TAG = WEIGHTS.removeprefix("densenet121-res224-")
OUTPUT_ROOT = RUN_ROOT / f"torchxrayvision_densenet121_{WEIGHT_TAG}"
TRAINING_SOURCE = {
    "densenet121-res224-mimic_ch": "MIMIC-CXR with CheXpert-labeler targets",
    "densenet121-res224-chex": "CheXpert (Stanford)",
}.get(WEIGHTS, "TorchXRayVision pretrained checkpoint")
TASKS = {
    "Pleural Effusion": "Effusion",
    "Cardiomegaly": "Cardiomegaly",
}
DEVICE = torch.device(os.environ.get("CXR_DEVICE", "tpu"))
BATCH_SIZE = int(os.environ.get("CXR_EVAL_BATCH_SIZE", "32"))
NUM_WORKERS = int(os.environ.get("CXR_NUM_WORKERS", "4"))
AMP = os.environ.get("CXR_TPU_AMP", "1") == "1"
COMPILE = os.environ.get("CXR_TPU_COMPILE", "1") == "1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def make_test_dataset(task: str) -> CXRDataset:
    return CXRDataset(
        csv_file_img=str(CSV_ROOT / "chexpert.sample.test.csv"),
        image_size=(224, 224),
        img_data_dir=IMAGE_ROOT,
        augmentation=False,
        pseudo_rgb=False,
        use_cache=False,
        nsamples=1,
        invariant_sampling=False,
        protected_race_set=[0, 1],
        protected_label="view_label",
        disease_name=task,
    )


def make_loader(dataset: CXRDataset) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        worker_init_fn=lambda _: torch.set_num_threads(1),
        persistent_workers=NUM_WORKERS > 0,
    )


def xrv_preprocess(images: torch.Tensor) -> torch.Tensor:
    """Normalize grayscale uint8-range images for XRV's expected input scale.

    The prepared CheXpert files are already square 224x224 images, so XRV's
    center crop and 224 resizer would be identity operations here.
    """
    image_array = images[:, :1].numpy()
    return torch.from_numpy(xrv.datasets.normalize(image_array, 255))


@torch.inference_mode()
def score_dataset(model: torch.nn.Module, dataset: CXRDataset, column: int) -> tuple[np.ndarray, np.ndarray]:
    all_labels: list[np.ndarray] = []
    all_scores: list[np.ndarray] = []
    loader = make_loader(dataset)
    total_batches = len(loader)
    for batch_idx, batch in enumerate(loader, start=1):
        x = xrv_preprocess(batch["image"])
        if DEVICE.type == "tpu" and AMP:
            with torch.autocast("tpu", dtype=torch.bfloat16):
                output = model(x.to(DEVICE))
        else:
            output = model(x.to(DEVICE))
        all_labels.append(batch["label"].numpy().astype(np.int64))
        all_scores.append(output[:, column].float().cpu().numpy())
        if batch_idx <= 3 or batch_idx % 10 == 0 or batch_idx == total_batches:
            print(json.dumps({"pathology": dataset.disease_name, "batch": batch_idx, "total_batches": total_batches}), flush=True)
    return np.concatenate(all_labels), np.concatenate(all_scores)


def main() -> None:
    if DEVICE.type == "tpu":
        import torch_tpu  # noqa: F401  # Register the TorchTPU device.

        if not torch.accelerator.is_available() or torch.accelerator.current_accelerator().type != "tpu":
            raise RuntimeError("CXR_DEVICE=tpu was requested, but TorchTPU reports no available TPU")
    torch.set_num_threads(32 if DEVICE.type == "tpu" else 4)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    cache_dir = OUTPUT_ROOT / "weights_cache"

    model = xrv.models.DenseNet(weights=WEIGHTS, cache_dir=str(cache_dir))
    # op_norm uses boolean indexed assignment (aten.index_put), which the
    # installed TorchTPU compiler cannot trace. Raw logits preserve ranking,
    # so they produce the same AUROC without this calibration transform.
    model.op_threshs = None
    model.apply_sigmoid = False
    pathology_names = list(model.pathologies)
    indices = {}
    for task, pathology in TASKS.items():
        if pathology not in pathology_names:
            raise RuntimeError(f"{WEIGHTS} does not provide a trained {pathology} output")
        indices[task] = pathology_names.index(pathology)
    model = model.to(DEVICE).eval()
    if DEVICE.type == "tpu" and COMPILE:
        # Inputs are pre-normalized and fixed at 224x224; skip XRV's one-time
        # Python range diagnostic so Dynamo can capture the model forward.
        xrv.utils.warning_log["norm_check"] = True
        print(json.dumps({"status": "compiling model", "backend": "tpu", "dynamic": False}), flush=True)
        model = torch.compile(model, backend="tpu", dynamic=False)
    weights_file = Path(model.weights_filename_local)

    resnet_summary = json.loads((RUN_ROOT / "summary.json").read_text())
    comparison: dict[str, dict] = {}
    for task, pathology in TASKS.items():
        dataset = make_test_dataset(task)
        labels, scores = score_dataset(model, dataset, indices[task])
        auc = float(roc_auc_score(labels, scores))
        predictions_path = OUTPUT_ROOT / task.lower().replace(" ", "_") / "predictions.test.csv"
        predictions_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"target": labels, "score": scores}).to_csv(predictions_path, index=False)

        resnet_test_auc = resnet_summary[task]["selected_top_five"][0]["test_auroc"]
        comparison[task] = {
            "xrv_pathology": pathology,
            "test_examples": int(len(labels)),
            "xrv_test_auroc": auc,
            "resnet18_seed43_test_auroc": float(resnet_test_auc),
            "xrv_minus_resnet18_auroc": auc - float(resnet_test_auc),
            "prediction_file": str(predictions_path.relative_to(REPO)),
        }
        print(json.dumps({"task": task, **comparison[task]}), flush=True)

    config = {
        "model": "TorchXRayVision DenseNet-121",
        "weights": WEIGHTS,
        "training_source": TRAINING_SOURCE,
        "input_resolution": 224,
        "device": str(DEVICE),
        "tpu_amp": AMP if DEVICE.type == "tpu" else False,
        "torch_compile": COMPILE if DEVICE.type == "tpu" else False,
        "torchxrayvision_version": importlib.metadata.version("torchxrayvision"),
        "weights_path": str(weights_file),
        "weights_sha256": sha256(weights_file),
        "test_split": "the same project CheXpert test CSV and binary cohort filter as the ResNet-18 run",
        "preprocessing": "grayscale, XRV normalize(maxval=255) to [-1024, 1024]; center crop and resize omitted because stored images are already square 224x224",
        "score": "raw classifier logits; the XRV sigmoid/operating-point transform is omitted because AUROC is rank-based",
        "tasks": comparison,
        "limitations": [
            "Architecture differs from the ResNet-18 baseline.",
            "The pretrained model uses a different training source from the ResNet-18 baseline; this is not a controlled architecture-only comparison.",
            "XRV preprocessing differs from the ResNet-18 run and follows the pretrained model's expected input convention.",
        ],
    }
    if WEIGHTS == "densenet121-res224-chex":
        config["limitations"].append(
            "Potential overlap between TorchXRayVision's CheXpert training images and this test split was not verified; interpret this result as exploratory."
        )
    (OUTPUT_ROOT / "comparison.json").write_text(json.dumps(config, indent=2) + "\n")


if __name__ == "__main__":
    main()
