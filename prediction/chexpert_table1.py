"""Training and evaluation helpers for the CheXpert-only Table 1 replication."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import auc, roc_auc_score, roc_curve
from torch.utils.data import DataLoader

from prediction.backbones import DenseNet, EfficientNetB0, ResNet18
from prediction.datasets.cxr import CXRDataset


MODEL_TYPES = {
    "resnet18": ResNet18,
    "densenet121": DenseNet,
    "efficientnetb0": EfficientNetB0,
}
TASKS = ("Pleural Effusion", "Cardiomegaly")
METHODS = ("baseline", "view", "race", "sex")


def _protected_set(method: str) -> list[int]:
    return {"race": [0, 1, 2], "view": [0, 1], "sex": [0, 1]}.get(method, [0, 1, 2])


def build_datasets(
    data_root: str | Path,
    task: str,
    method: str,
    nsamples: int = 2,
) -> tuple[CXRDataset, CXRDataset, CXRDataset]:
    root = Path(data_root)
    csv_dir = root / "datafiles" / "chexpert"
    image_root = str(root) + os.sep
    protected_label = method if method != "baseline" else "sex"
    protected_label += "_label"
    protected_set = _protected_set(method)

    def make(split: str, augment: bool, sample_groups: bool) -> CXRDataset:
        return CXRDataset(
            csv_file_img=str(csv_dir / f"chexpert.sample.{split}.csv"),
            image_size=(224, 224),
            img_data_dir=image_root,
            augmentation=augment,
            pseudo_rgb=True,
            use_cache=False,
            nsamples=nsamples if sample_groups else 1,
            invariant_sampling=sample_groups,
            protected_race_set=protected_set,
            protected_label=protected_label,
            disease_name=task,
        )

    is_invariant = method != "baseline"
    return (
        make("train", True, is_invariant),
        make("val", False, False),
        make("test", False, False),
    )


def make_loader(
    dataset: CXRDataset,
    batch_size: int,
    workers: int,
    shuffle: bool = False,
    seed: int = 42,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        drop_last=shuffle,
        pin_memory=False,
        generator=generator,
        persistent_workers=workers > 0,
    )


def _move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device=device) for key, value in batch.items()}


def _evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    include_predictions: bool = False,
) -> dict[str, Any]:
    model.eval()
    logits_list: list[np.ndarray] = []
    scores: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            batch = _move_batch(batch, device)
            _, logits = model(batch["image"])
            flat_logits = logits.reshape(-1)
            score = torch.sigmoid(flat_logits)
            logits_list.append(flat_logits.cpu().numpy())
            scores.append(score.cpu().numpy())
            targets.append(batch["label"].reshape(-1).cpu().numpy())
    y_logits = np.concatenate(logits_list)
    y_score = np.concatenate(scores)
    y_true = np.concatenate(targets).astype(np.int64)
    if np.unique(y_true).size != 2:
        raise ValueError("AUROC evaluation requires both classes")
    result: dict[str, Any] = {
        "auroc": float(roc_auc_score(y_true, y_score)),
        "n": int(len(y_true)),
    }
    if include_predictions:
        result.update({"targets": y_true, "logits": y_logits, "scores": y_score})
    return result


def _save_checkpoint(path: Path, model: torch.nn.Module, fields: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    tmp = path.with_suffix(".tmp")
    torch.save({"state_dict": state, **fields}, tmp)
    os.chmod(tmp, 0o600)
    os.replace(tmp, path)


def create_model(model_name: str, pretrained: bool = True) -> torch.nn.Module:
    if model_name not in MODEL_TYPES:
        raise ValueError(f"Unsupported backbone: {model_name}")
    # The existing repository backbones use torchvision ImageNet-1K weights.
    # Keep that base behavior; the flag is retained for bounded unit tests.
    if not pretrained:
        raise ValueError("The repository backbones currently require ImageNet-1K weights")
    return MODEL_TYPES[model_name](
        num_classes=1,
        inv_loss_coefficient=1.0,
        lr_backbone=0.001,
        lr_disease=0.0001,
    )


def train_one(
    *,
    data_root: str | Path,
    output_dir: str | Path,
    task: str,
    model_name: str,
    method: str,
    seed: int,
    epochs: int = 20,
    batch_size: int = 32,
    nsamples: int = 2,
    num_workers: int = 4,
    device_name: str = "tpu",
    max_steps: int | None = None,
    pretrained: bool = True,
    datasets: tuple[CXRDataset, CXRDataset, CXRDataset] | None = None,
    runtime_provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device(device_name)
    # Fail early rather than silently falling back to CPU.
    torch.empty(1, device=device)

    if datasets is None:
        datasets = build_datasets(data_root, task, method, nsamples)
    train_set, val_set, _ = datasets
    train_loader = make_loader(
        train_set, batch_size, num_workers, shuffle=True, seed=seed
    )
    val_loader = make_loader(val_set, batch_size, num_workers, seed=seed)

    model = create_model(model_name, pretrained=pretrained).to(device)
    optimizer = torch.optim.Adam(
        [
            {"params": model.backbone.parameters(), "lr": 0.001},
            {"params": model.fc_disease.parameters(), "lr": 0.0001},
        ]
    )
    run_dir = Path(output_dir) / task.lower().replace(" ", "_") / model_name / method / f"seed-{seed}"
    run_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    run_dir.chmod(0o700)
    invariant = method != "baseline"
    config = {
        "task": task,
        "model": model_name,
        "method": method,
        "seed": seed,
        "epochs_requested": epochs,
        "batch_size": batch_size,
        "batch_size_semantics": "invariant groups" if invariant else "images",
        "invariant_samples_per_group": nsamples if invariant else 1,
        "effective_images_per_optimizer_step": batch_size * (nsamples if invariant else 1),
        "optimizer": "Adam",
        "lr_backbone": 0.001,
        "lr_classifier": 0.0001,
        "loss": (
            "BCEWithLogitsLoss + repository invariant objective" if invariant
            else "BCEWithLogitsLoss"
        ),
        "image_size": [224, 224],
        "pretrained": "ImageNet-1K",
        "checkpoint_selection": "maximum validation AUROC within seed",
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    os.chmod(run_dir / "config.json", 0o600)
    if runtime_provenance is not None:
        (run_dir / "runtime.json").write_text(json.dumps(runtime_provenance, indent=2) + "\n")
        os.chmod(run_dir / "runtime.json", 0o600)
    checkpoint_path = run_dir / "best.ckpt"
    best_auc = -float("inf")
    best_epoch = -1
    history: list[dict[str, float | int]] = []
    step_count = 0

    for epoch in range(epochs):
        model.train()
        loss_sum = torch.zeros((), device=device)
        batches = 0
        for batch in train_loader:
            batch = _move_batch(batch, device)
            optimizer.zero_grad()
            if batch["image"].ndim == 5:
                loss_class, loss_inv = model.process_batch_list(batch)
            else:
                loss_class, loss_inv = model.process_batch(batch)
            loss = loss_class + loss_inv
            loss.backward()
            optimizer.step()
            loss_sum += loss.detach()
            batches += 1
            step_count += 1
            if max_steps is not None and step_count >= max_steps:
                break
        if batches == 0:
            raise ValueError("Training loader produced no full batches")

        val = _evaluate(model, val_loader, device)
        mean_loss = float((loss_sum / batches).cpu().item())
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": mean_loss,
                "val_auroc": val["auroc"],
                "val_n": val["n"],
                "steps": step_count,
            }
        )
        if val["auroc"] > best_auc:
            best_auc = val["auroc"]
            best_epoch = epoch + 1
            _save_checkpoint(
                checkpoint_path,
                model,
                {
                    "epoch": best_epoch,
                    "best_val_auroc": best_auc,
                    "seed": seed,
                    "task": task,
                    "model": model_name,
                    "method": method,
                    "config": config,
                },
            )

        if max_steps is not None and step_count >= max_steps:
            break

    with (run_dir / "history.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    digest = hashlib.sha256()
    with checkpoint_path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    result = {
        "config": config,
        "runtime_provenance": runtime_provenance,
        "checkpoint_sha256": digest.hexdigest(),
        "task": task,
        "model": model_name,
        "method": method,
        "seed": seed,
        "best_epoch": best_epoch,
        "best_val_auroc": best_auc,
        "epochs_completed": len(history),
        "steps": step_count,
        "checkpoint": str(checkpoint_path),
        "history": history,
    }
    (run_dir / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    os.chmod(run_dir / "metrics.json", 0o600)
    os.chmod(run_dir / "history.csv", 0o600)
    return result


def evaluate_checkpoint(
    checkpoint: str | Path,
    *,
    data_root: str | Path,
    task: str,
    model_name: str,
    method: str,
    batch_size: int = 32,
    nsamples: int = 2,
    num_workers: int = 4,
    device_name: str = "tpu",
    datasets: tuple[CXRDataset, CXRDataset, CXRDataset] | None = None,
    predictions_path: str | Path | None = None,
) -> dict[str, Any]:
    if datasets is None:
        datasets = build_datasets(data_root, task, method, nsamples)
    _, _, test_set = datasets
    test_loader = make_loader(test_set, batch_size, num_workers)
    device = torch.device(device_name)
    model = create_model(model_name).to(device)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(payload["state_dict"])
    model.to(device)
    result = _evaluate(model, test_loader, device, include_predictions=True)
    y_true = result.pop("targets")
    y_logits = result.pop("logits")
    y_score = result.pop("scores")
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_curve_auc = float(auc(fpr, tpr))
    if not np.isclose(result["auroc"], roc_curve_auc, rtol=0, atol=1e-12):
        raise AssertionError("roc_auc_score and trapezoidal ROC AUC disagree")
    result["roc_curve_auc_check"] = roc_curve_auc
    if predictions_path is not None:
        output_path = Path(predictions_path)
        output_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        with output_path.open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(("target", "logit", "positive_score"))
            writer.writerows(zip(y_true.tolist(), y_logits.tolist(), y_score.tolist()))
        os.chmod(output_path, 0o600)
        result["predictions_path"] = str(output_path)
    result.update(
        {
            "task": task,
            "model": model_name,
            "method": method,
            "seed": int(payload["seed"]),
            "checkpoint": str(checkpoint),
        }
    )
    return result
