#!/usr/bin/env python3
"""Evaluate the CheXpert seed-43 ResNet-18 checkpoints on CXR-RAIT."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pydicom
import torch
from pydicom.pixels import apply_modality_lut, apply_voi_lut
from skimage.io import imread, imsave
from skimage.transform import resize
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models

warnings.filterwarnings("ignore", message="Unknown encoding.*")


REPO = Path(__file__).resolve().parents[1]
RUN_ROOT = REPO / "chexpert/runs/resnet18_baseline_tpu_ce"
TASKS = {
    "pleural_effusion": ("pleural_effusion", "Pleural Effusion"),
    "cardiomegaly": ("cardiomegaly", "Cardiomegaly"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dicom_to_uint8(path: Path) -> np.ndarray:
    """Render a DICOM grayscale image using its modality/VOI metadata."""
    ds = pydicom.dcmread(path)
    pixels = np.asarray(ds.pixel_array)
    if pixels.ndim != 2:
        raise ValueError(f"Expected one 2D image in {path.name}; got {pixels.shape}")

    pixels = apply_modality_lut(pixels, ds)
    has_voi = "VOILUTSequence" in ds or ("WindowCenter" in ds and "WindowWidth" in ds)
    if has_voi:
        pixels = apply_voi_lut(pixels, ds)

    pixels = np.asarray(pixels, dtype=np.float32)
    finite = np.isfinite(pixels)
    if not finite.any():
        raise ValueError(f"DICOM has no finite pixel values: {path.name}")
    low = float(pixels[finite].min())
    high = float(pixels[finite].max())
    if high <= low:
        rendered = np.zeros(pixels.shape, dtype=np.uint8)
    else:
        rendered = np.clip((pixels - low) * (255.0 / (high - low)), 0, 255).astype(np.uint8)

    if getattr(ds, "PhotometricInterpretation", "MONOCHROME2") == "MONOCHROME1":
        rendered = 255 - rendered
    return resize(rendered, output_shape=(224, 224), preserve_range=True).astype(np.uint8)


def decode_job(job: tuple[str, Path, Path]) -> tuple[str, Path, np.ndarray]:
    cv_filename, source, destination = job
    return cv_filename, destination, dicom_to_uint8(source)


def prepare_images(
    rows: list[dict[str, str]], dicom_root: Path, image_root: Path, workers: int
) -> dict[str, Path]:
    image_root.mkdir(parents=True, exist_ok=True)
    dicom_by_name: dict[str, Path] = {}
    for path in dicom_root.rglob("*.dcm"):
        if path.name in dicom_by_name:
            raise RuntimeError(f"Duplicate DICOM basename in staging area: {path.name}")
        dicom_by_name[path.name] = path

    resolved: dict[str, Path] = {}
    jobs: list[tuple[str, Path, Path]] = []
    for row in rows:
        cv_filename = row["cv_filename"].strip()
        image_name = f"{Path(cv_filename).stem}.dcm"
        source = dicom_by_name.get(image_name)
        if source is None:
            raise FileNotFoundError(f"No staged DICOM matches {cv_filename}")
        destination = image_root / f"{Path(cv_filename).stem}.png"
        resolved[cv_filename] = destination
        if not destination.exists():
            jobs.append((cv_filename, source, destination))

    completed = len(rows) - len(jobs)
    if completed:
        print(json.dumps({"stage": "preprocessing", "completed": completed, "total": len(rows), "reused": completed}), flush=True)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for _, destination, image in pool.map(decode_job, jobs, chunksize=4):
            imsave(destination, image, check_contrast=False)
            completed += 1
            if completed % 100 == 0 or completed == len(rows):
                print(json.dumps({"stage": "preprocessing", "completed": completed, "total": len(rows)}), flush=True)
    if len(resolved) != len(rows):
        raise RuntimeError("The evaluation CSV contains duplicate cv_filename values")
    return resolved


class RaitImageDataset(Dataset):
    def __init__(self, rows: list[dict[str, str]], image_paths: dict[str, Path], label: str):
        self.rows = rows
        self.image_paths = image_paths
        self.label = label

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        row = self.rows[index]
        image = imread(self.image_paths[row["cv_filename"]]).astype(np.float32)
        image = np.repeat(image[None, :, :], 3, axis=0)
        return torch.from_numpy(image), int(row[self.label])


def load_task_model(task_dir: Path, device: torch.device) -> tuple[nn.Module, nn.Module, Path]:
    checkpoint_path = task_dir / "seed_43/best.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    backbone = models.resnet18(weights=None)
    backbone.fc = nn.Identity()
    classifier = nn.Linear(512, 2)
    backbone.load_state_dict(checkpoint["state"]["backbone"])
    classifier.load_state_dict(checkpoint["state"]["classifier"])
    backbone.to(device).eval()
    classifier.to(device).eval()
    return backbone, classifier, checkpoint_path


@torch.inference_mode()
def evaluate_task(
    task: str,
    task_info: tuple[str, str],
    rows: list[dict[str, str]],
    image_paths: dict[str, Path],
    output_root: Path,
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> dict:
    label_col, model_task = task_info
    task_dir = RUN_ROOT / task
    backbone, classifier, checkpoint_path = load_task_model(task_dir, device)
    dataset = RaitImageDataset(rows, image_paths, label_col)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    total_batches = len(loader)

    targets: list[np.ndarray] = []
    scores: list[np.ndarray] = []
    for batch_index, (images, labels) in enumerate(loader, start=1):
        if device.type == "tpu":
            with torch.autocast("tpu", dtype=torch.bfloat16):
                logits = classifier(backbone(images.to(device)))
        else:
            logits = classifier(backbone(images.to(device)))
        scores.append(torch.softmax(logits.float(), dim=1)[:, 1].cpu().numpy())
        targets.append(labels.numpy().astype(np.int64))
        if batch_index % 20 == 0 or batch_index == total_batches:
            print(json.dumps({"stage": "inference", "task": model_task, "batch": batch_index, "total_batches": total_batches}), flush=True)

    y = np.concatenate(targets)
    score = np.concatenate(scores)
    if len(np.unique(y)) != 2:
        raise RuntimeError(f"{model_task} target has only one class; AUROC cannot be computed")
    task_output = output_root / task
    task_output.mkdir(parents=True, exist_ok=True)
    prediction_path = task_output / "predictions.csv"
    with prediction_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["cv_filename", "target", "score_positive"])
        writer.writeheader()
        writer.writerows(
            {"cv_filename": row["cv_filename"], "target": int(target), "score_positive": float(value)}
            for row, target, value in zip(rows, y, score, strict=True)
        )

    return {
        "task": model_task,
        "checkpoint": str(checkpoint_path.relative_to(REPO)),
        "checkpoint_sha256": sha256(checkpoint_path),
        "examples": int(len(y)),
        "positive_examples": int(y.sum()),
        "negative_examples": int(len(y) - y.sum()),
        "auroc": float(roc_auc_score(y, score)),
        "average_precision": float(average_precision_score(y, score)),
        "predictions": str(prediction_path.relative_to(REPO)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, default=RUN_ROOT / "cxr_rait/labels.csv")
    parser.add_argument("--dicom-root", type=Path, default=RUN_ROOT / "cxr_rait/dicom")
    parser.add_argument("--image-root", type=Path, default=RUN_ROOT / "cxr_rait/preprocessed_224")
    parser.add_argument("--output-root", type=Path, default=RUN_ROOT / "cxr_rait/results")
    parser.add_argument("--device", default=os.environ.get("CXR_DEVICE", "cpu"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--preprocess-workers", type=int, default=4)
    args = parser.parse_args()
    args.output_root = args.output_root.resolve()

    device = torch.device(args.device)
    if device.type == "tpu":
        import torch_tpu  # noqa: F401  # Register the TorchTPU device.

        if not torch.accelerator.is_available() or torch.accelerator.current_accelerator().type != "tpu":
            raise RuntimeError("CXR_DEVICE=tpu was requested, but TorchTPU reports no available TPU")
    torch.set_num_threads(4 if device.type == "cpu" else 32)
    with args.labels.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    required = {"cv_filename", "pleural_effusion", "cardiomegaly"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"Labels CSV must contain {sorted(required)}")
    if len(rows) != 3248:
        raise ValueError(f"Expected 3,248 image-backed rows, found {len(rows)}")
    if len({row["cv_filename"] for row in rows}) != len(rows):
        raise ValueError("Labels CSV contains duplicate cv_filename values")
    for row in rows:
        for label, _ in TASKS.values():
            if row[label] not in {"0", "1"}:
                raise ValueError(f"Invalid {label} label for {row['cv_filename']}: {row[label]!r}")

    image_paths = prepare_images(rows, args.dicom_root, args.image_root, args.preprocess_workers)
    args.output_root.mkdir(parents=True, exist_ok=True)
    results = {
        task: evaluate_task(
            task,
            task_info,
            rows,
            image_paths,
            args.output_root,
            device,
            args.batch_size,
            args.num_workers,
        )
        for task, task_info in TASKS.items()
    }
    summary = {
        "evaluation": "CXR-RAIT cross-domain evaluation of CheXpert-trained ResNet-18",
        "source_labels": str(args.labels),
        "source_labels_sha256": sha256(args.labels),
        "samples": len(rows),
        "device": str(device),
        "tpu_amp": device.type == "tpu",
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pydicom_version": importlib.metadata.version("pydicom"),
        "preprocessing": "DICOM modality LUT then VOI LUT/window when present; MONOCHROME1 inversion; min-max uint8 rendering; resize to 224x224; repeat grayscale into 3 channels; retain 0-255 intensity scale without ImageNet normalization.",
        "label_interpretation": "Binary targets from the supplied CXR-RAIT CSV; 0 is below the prediction_nlp threshold and is treated as the negative/No Finding proxy, not a verified normal radiograph.",
        "metrics": "AUROC and average precision from positive-class softmax scores; no target-domain threshold tuning.",
        "tasks": results,
    }
    summary_path = args.output_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"summary": str(summary_path.relative_to(REPO)), "tasks": results}, indent=2), flush=True)


if __name__ == "__main__":
    main()
