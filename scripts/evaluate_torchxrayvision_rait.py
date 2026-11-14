#!/usr/bin/env python3
"""Evaluate TorchXRayVision DenseNet-121 checkpoints on the CXR-RAIT cohort."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np
import torch
from skimage.io import imread
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset
import torchxrayvision as xrv


REPO = Path(__file__).resolve().parents[1]
RUN_ROOT = REPO / "chexpert/runs/resnet18_baseline_tpu_ce/cxr_rait"
XRV_RUN_ROOT = REPO / "chexpert/runs/resnet18_baseline_tpu_ce"
WEIGHTS = {
    "chexpert": "densenet121-res224-chex",
    "mimic_ch": "densenet121-res224-mimic_ch",
}
LABELS = ("pleural_effusion", "cardiomegaly")
PATHOLOGIES = {"pleural_effusion": "Effusion", "cardiomegaly": "Cardiomegaly"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class RaitDataset(Dataset):
    def __init__(self, rows: list[dict[str, str]], image_root: Path):
        self.rows = rows
        self.image_root = image_root

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        row = self.rows[index]
        image = imread(self.image_root / f"{Path(row['cv_filename']).stem}.png")
        if image.ndim == 3:
            image = image[..., 0]
        image = image.astype(np.float32, copy=False)[None, :, :]
        targets = np.asarray([int(row[label]) for label in LABELS], dtype=np.int64)
        return image, targets


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, default=RUN_ROOT / "labels.csv")
    parser.add_argument("--image-root", type=Path, default=RUN_ROOT / "preprocessed_224")
    parser.add_argument("--output-root", type=Path, default=RUN_ROOT / "xrv_results")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--weights", choices=["all", *WEIGHTS], default="all")
    parser.add_argument("--compile-model", action="store_true", help="Compile fixed-shape inference with TorchTPU")
    args = parser.parse_args()
    args.output_root = args.output_root.resolve()

    device = torch.device(args.device)
    if device.type == "tpu":
        import torch_tpu  # noqa: F401  # Register the TorchTPU device.

        if not torch.accelerator.is_available() or torch.accelerator.current_accelerator().type != "tpu":
            raise RuntimeError("--device tpu was requested, but TorchTPU reports no available TPU")
    if args.compile_model and device.type != "tpu":
        raise ValueError("--compile-model requires --device tpu")
    torch.set_num_threads(4 if device.type == "cpu" else 32)
    with args.labels.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    required = {"cv_filename", *LABELS}
    if len(rows) != 3248 or not required.issubset(rows[0]):
        raise ValueError(f"Expected 3,248 rows and columns {sorted(required)}")
    if len({row["cv_filename"] for row in rows}) != len(rows):
        raise ValueError("Labels CSV contains duplicate cv_filename values")
    for row in rows:
        if any(row[label] not in {"0", "1"} for label in LABELS):
            raise ValueError(f"Non-binary label in row {row['cv_filename']}")

    dataset = RaitDataset(rows, args.image_root)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    selected = WEIGHTS if args.weights == "all" else {args.weights: WEIGHTS[args.weights]}
    args.output_root.mkdir(parents=True, exist_ok=True)
    results = {}

    for source, weight_name in selected.items():
        run_dir = args.output_root / source
        run_dir.mkdir(parents=True, exist_ok=True)
        cache_tag = {"chexpert": "chex", "mimic_ch": "mimic_ch"}[source]
        cache_source = XRV_RUN_ROOT / f"torchxrayvision_densenet121_{cache_tag}" / "weights_cache"
        model = xrv.models.DenseNet(weights=weight_name, cache_dir=str(cache_source))
        model.op_threshs = None
        model.apply_sigmoid = False
        pathologies = list(model.pathologies)
        columns = [pathologies.index(PATHOLOGIES[label]) for label in LABELS]
        model = model.to(device).eval()
        if args.compile_model:
            # XRV's range diagnostic uses Python-side checks Dynamo cannot trace.
            xrv.utils.warning_log["norm_check"] = True
            model = torch.compile(model, backend="tpu", dynamic=False)
        weights_file = Path(model.weights_filename_local)
        all_targets: list[np.ndarray] = []
        all_scores: list[np.ndarray] = []

        with torch.inference_mode():
            for batch_index, (image, target) in enumerate(loader, start=1):
                x = xrv.datasets.normalize(image.numpy(), 255)
                x = torch.from_numpy(x).to(device)
                actual_batch_size = len(target)
                if device.type == "tpu" and actual_batch_size < args.batch_size:
                    padding = x[:1].expand(args.batch_size - actual_batch_size, *x.shape[1:])
                    x = torch.cat((x, padding), dim=0)
                if device.type == "tpu":
                    with torch.autocast("tpu", dtype=torch.bfloat16):
                        output = model(x)
                else:
                    output = model(x)
                scores = output[:actual_batch_size, columns].float().cpu().numpy()
                all_targets.append(target.numpy())
                all_scores.append(scores)
                if batch_index % 20 == 0 or batch_index == len(loader):
                    print(json.dumps({"weights": weight_name, "batch": batch_index, "batches": len(loader)}), flush=True)

        y = np.concatenate(all_targets)
        score = np.concatenate(all_scores)
        task_metrics = {}
        for index, label in enumerate(LABELS):
            task_targets = y[:, index]
            task_scores = score[:, index]
            if len(np.unique(task_targets)) != 2:
                raise ValueError(f"{label} target has only one class")
            predictions_path = run_dir / label / "predictions.csv"
            predictions_path.parent.mkdir(parents=True, exist_ok=True)
            with predictions_path.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=["cv_filename", "target", "score"])
                writer.writeheader()
                writer.writerows(
                    {"cv_filename": row["cv_filename"], "target": int(target), "score": float(value)}
                    for row, target, value in zip(rows, task_targets, task_scores, strict=True)
                )
            task_metrics[label] = {
                "pathology_output": PATHOLOGIES[label],
                "examples": int(len(task_targets)),
                "positive_examples": int(task_targets.sum()),
                "negative_examples": int(len(task_targets) - task_targets.sum()),
                "auroc": float(roc_auc_score(task_targets, task_scores)),
                "average_precision": float(average_precision_score(task_targets, task_scores)),
                "predictions": str(predictions_path.relative_to(REPO)),
            }

        results[source] = {
            "model": "TorchXRayVision DenseNet-121",
            "weights": weight_name,
            "training_source": {
                "chexpert": "CheXpert (Stanford)",
                "mimic_ch": "MIMIC-CXR with CheXpert-labeler targets",
            }[source],
            "weights_path": str(weights_file),
            "weights_sha256": sha256(weights_file),
            "torchxrayvision_version": importlib.metadata.version("torchxrayvision"),
            "torch_version": torch.__version__,
            "device": str(device),
            "tpu_amp": device.type == "tpu",
            "torch_compile": args.compile_model,
            "batch_size": args.batch_size,
            "samples": len(rows),
            "tasks": task_metrics,
        }

    summary = {
        "evaluation": "TorchXRayVision DenseNet-121 cross-domain evaluation on CXR-RAIT",
        "labels_csv": str(args.labels),
        "labels_csv_sha256": sha256(args.labels),
        "image_preprocessing": "Previously rendered CXR-RAIT DICOMs: DICOM modality LUT and VOI LUT/window where present; MONOCHROME1 inversion; min-max uint8 rendering; resize 224x224. XRV grayscale normalization maps 8-bit values to [-1024, 1024].",
        "score": "Raw pathology logits; sigmoid/operating-point calibration omitted because AUROC is rank based.",
        "label_interpretation": "Supplied CXR-RAIT binary labels; 0 is below the prediction_nlp threshold and serves as a negative/No Finding proxy, not verified normality.",
        "metrics": "AUROC and average precision; no threshold tuning on target domain.",
        "models": results,
    }
    summary_path = args.output_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"summary": str(summary_path.relative_to(REPO)), "models": results}, indent=2), flush=True)


if __name__ == "__main__":
    main()
