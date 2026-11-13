from __future__ import annotations

import json
import os
import random
from pathlib import Path

os.environ.setdefault("TORCH_HOME", str(Path(__file__).resolve().parent / "torch-cache"))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from torchvision import models

from prediction.datasets.cxr import CXRDataset


REPO = Path(__file__).resolve().parents[1]
RUN_ROOT = Path(__file__).resolve().parent / "runs" / "resnet18_baseline_tpu_ce"
DATA_ROOT = "chexpert/hf_preprocessed/"
CSV_ROOT = REPO / "datafiles" / "chexpert"
SEEDS = tuple(int(s) for s in os.environ.get("CXR_SEEDS", ",".join(map(str, range(42, 52)))).split(",") if s.strip())
TASKS = tuple(s.strip() for s in os.environ.get("CXR_TASKS", "Pleural Effusion,Cardiomegaly").split(",") if s.strip())
BATCH_SIZE = 32
EPOCHS = 20
LR_BACKBONE = 1e-3
LR_CLASSIFIER = 1e-4
NUM_WORKERS = int(os.environ.get("CXR_NUM_WORKERS", "0"))
DEVICE = torch.device(os.environ.get("CXR_DEVICE", "tpu"))
AMP = os.environ.get("CXR_TPU_AMP", "1") == "1"


def worker_init(_: int) -> None:
    torch.set_num_threads(1)


def make_model() -> tuple[nn.Module, nn.Module]:
    backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    n_features = backbone.fc.in_features
    backbone.fc = nn.Identity()
    classifier = nn.Linear(n_features, 2)
    return backbone, classifier


def model_context():
    if DEVICE.type == "tpu" and AMP:
        return torch.autocast(device_type="tpu", dtype=torch.bfloat16)
    return torch.autocast(device_type=DEVICE.type, enabled=False)


def install_tpu_batchnorm_stats_fix(model: nn.Module) -> int:
    """Restore PyTorch BatchNorm EMA buffers on TorchTPU builds that omit updates."""
    if DEVICE.type != "tpu":
        return 0

    def update_running_stats(module: nn.modules.batchnorm._BatchNorm, args: tuple[torch.Tensor, ...]) -> None:
        if not module.training or not module.track_running_stats or module.running_mean is None:
            return
        x = args[0].detach().float()
        reduction_dims = (0, *range(2, x.ndim))
        sample_count = x.numel() // x.shape[1]
        if sample_count < 2:
            return
        mean = x.mean(dim=reduction_dims)
        var = x.var(dim=reduction_dims, correction=1)
        with torch.no_grad():
            module.num_batches_tracked.add_(1)
            factor = module.momentum
            if factor is None:
                factor = 1.0 / float(module.num_batches_tracked.item())
            module.running_mean.mul_(1.0 - factor).add_(mean, alpha=factor)
            module.running_var.mul_(1.0 - factor).add_(var, alpha=factor)

    batch_norms = [m for m in model.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    for layer in batch_norms:
        layer.register_forward_pre_hook(update_running_stats)
    return len(batch_norms)


def make_dataset(split: str, task: str, augmentation: bool) -> CXRDataset:
    return CXRDataset(
        csv_file_img=str(CSV_ROOT / f"chexpert.sample.{split}.csv"),
        image_size=(224, 224),
        img_data_dir=DATA_ROOT,
        augmentation=augmentation,
        pseudo_rgb=True,
        use_cache=False,
        nsamples=1,
        invariant_sampling=False,
        protected_race_set=[0, 1],
        protected_label="view_label",
        disease_name=task,
    )


def make_loader(dataset: CXRDataset, *, train: bool, seed: int) -> DataLoader:
    kwargs = {
        "dataset": dataset,
        "batch_size": BATCH_SIZE,
        "shuffle": train,
        "num_workers": NUM_WORKERS,
        "drop_last": train,
        "worker_init_fn": worker_init,
        "persistent_workers": NUM_WORKERS > 0,
        "generator": torch.Generator().manual_seed(seed),
    }
    if NUM_WORKERS:
        kwargs["prefetch_factor"] = 2
    return DataLoader(**kwargs)


@torch.no_grad()
def predict(backbone: nn.Module, classifier: nn.Module, loader: DataLoader) -> tuple[np.ndarray, np.ndarray]:
    backbone.eval()
    classifier.eval()
    ys: list[np.ndarray] = []
    scores: list[np.ndarray] = []
    for batch in loader:
        x = batch["image"].to(DEVICE)
        y = batch["label"].float().to(DEVICE)
        with model_context():
            z = backbone(x)
            logits = classifier(z)
        ys.append(y.cpu().numpy())
        scores.append(torch.softmax(logits.float(), dim=1)[:, 1].cpu().numpy())
    return np.concatenate(ys), np.concatenate(scores)


def save_predictions(path: Path, y: np.ndarray, score: np.ndarray, seed: int, task: str, split: str) -> None:
    pd.DataFrame({
        "target": y.astype(np.int8),
        "score_positive": score.astype(np.float32),
        "seed": seed,
        "task": task,
        "split": split,
    }).to_csv(path, index=False)


def checkpoint_state(backbone: nn.Module, classifier: nn.Module) -> dict[str, dict[str, torch.Tensor]]:
    return {
        "backbone": {k: v.detach().cpu().clone() for k, v in backbone.state_dict().items()},
        "classifier": {k: v.detach().cpu().clone() for k, v in classifier.state_dict().items()},
    }


def train_seed(task: str, seed: int, train_ds: CXRDataset, val_ds: CXRDataset, task_dir: Path) -> dict:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(32 if DEVICE.type == "tpu" else 4)
    backbone, classifier = make_model()
    install_tpu_batchnorm_stats_fix(backbone)
    backbone.to(DEVICE)
    classifier.to(DEVICE)
    optimizer = torch.optim.Adam([
        {"params": backbone.parameters(), "lr": LR_BACKBONE},
        {"params": classifier.parameters(), "lr": LR_CLASSIFIER},
    ])
    criterion = nn.CrossEntropyLoss()
    train_loader = make_loader(train_ds, train=True, seed=seed)
    val_loader = make_loader(val_ds, train=False, seed=seed)
    run_dir = task_dir / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    best_auc = -1.0
    best_epoch = 0
    best_state = None
    best_val: tuple[np.ndarray, np.ndarray] | None = None
    history: list[dict] = []
    for epoch in range(1, EPOCHS + 1):
        backbone.train()
        classifier.train()
        loss_sum = 0.0
        seen = 0
        for batch in train_loader:
            x = batch["image"].to(DEVICE)
            y = batch["label"].long().to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            with model_context():
                logits = classifier(backbone(x))
                loss = criterion(logits.float(), y)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach()) * len(y)
            seen += len(y)

        y_val, p_val = predict(backbone, classifier, val_loader)
        val_auc = float(roc_auc_score(y_val, p_val))
        row = {
            "epoch": epoch,
            "train_cross_entropy": loss_sum / seen,
            "val_auroc": val_auc,
            "train_examples": seen,
            "validation_examples": int(len(y_val)),
        }
        history.append(row)
        print(json.dumps({"task": task, "seed": seed, **row}), flush=True)
        if val_auc > best_auc:
            best_auc = val_auc
            best_epoch = epoch
            best_state = checkpoint_state(backbone, classifier)
            best_val = (y_val.copy(), p_val.copy())

    if best_state is None or best_val is None:
        raise RuntimeError(f"No best checkpoint produced for {task}, seed {seed}")
    torch.save({"state": best_state, "epoch": best_epoch, "seed": seed, "task": task}, run_dir / "best.pt")
    (run_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n")
    save_predictions(run_dir / "predictions.val_best.csv", *best_val, seed, task, "val")
    return {"seed": seed, "best_epoch": best_epoch, "best_val_auroc": best_auc, "run_dir": str(run_dir)}


def load_completed_seed(task: str, seed: int, task_dir: Path) -> dict | None:
    """Reuse a finished seed so narrowing a run does not repeat training."""
    run_dir = task_dir / f"seed_{seed}"
    history_path = run_dir / "history.json"
    checkpoint_path = run_dir / "best.pt"
    val_predictions_path = run_dir / "predictions.val_best.csv"
    if not all(path.exists() for path in (history_path, checkpoint_path, val_predictions_path)):
        return None
    history = json.loads(history_path.read_text())
    best = max(history, key=lambda row: row["val_auroc"])
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("task") != task or int(checkpoint.get("seed", -1)) != seed:
        raise RuntimeError(f"Cached checkpoint metadata mismatch for {task}, seed {seed}")
    return {
        "seed": seed,
        "best_epoch": int(best["epoch"]),
        "best_val_auroc": float(best["val_auroc"]),
        "run_dir": str(run_dir),
    }


def evaluate_top_seed(task: str, item: dict, test_ds: CXRDataset, task_dir: Path) -> dict:
    backbone, classifier = make_model()
    checkpoint = torch.load(Path(item["run_dir"]) / "best.pt", map_location="cpu", weights_only=True)
    backbone.load_state_dict(checkpoint["state"]["backbone"])
    classifier.load_state_dict(checkpoint["state"]["classifier"])
    backbone.to(DEVICE)
    classifier.to(DEVICE)
    loader = make_loader(test_ds, train=False, seed=item["seed"])
    y_test, p_test = predict(backbone, classifier, loader)
    auc = float(roc_auc_score(y_test, p_test))
    out = Path(item["run_dir"]) / "predictions.test.csv"
    save_predictions(out, y_test, p_test, item["seed"], task, "test")
    return {"seed": item["seed"], "best_epoch": item["best_epoch"], "val_auroc": item["best_val_auroc"], "test_auroc": auc, "test_examples": int(len(y_test))}


def main() -> None:
    if DEVICE.type == "tpu":
        if not torch.accelerator.is_available() or torch.accelerator.current_accelerator().type != "tpu":
            raise RuntimeError("CXR_DEVICE=tpu was requested, but TorchTPU reports no available TPU")
        print(json.dumps({"accelerator": "tpu", "device_count": torch.accelerator.device_count(), "amp": AMP}), flush=True)
    torch.set_num_threads(32 if DEVICE.type == "tpu" else 4)
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    config = {
        "source": "Hugging Face danjacobellis/chexpert, train split",
        "backbone": "ResNet-18, torchvision IMAGENET1K_V1 initialization",
        "tasks": list(TASKS),
        "method": "ordinary binary classification baseline",
        "seed_set": list(SEEDS),
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "optimizer": "Adam",
        "learning_rate_backbone": LR_BACKBONE,
        "learning_rate_classifier": LR_CLASSIFIER,
        "loss": "CrossEntropyLoss (repository ResNet baseline head: two logits)",
        "selection": "best epoch by validation AUROC; evaluate up to five highest-validation-AUROC configured seeds",
        "image_size": [224, 224],
        "test_policy": "evaluate configured seeds after validation ranking; a one-seed run evaluates that seed once",
        "device": str(DEVICE),
        "tpu_amp": AMP if DEVICE.type == "tpu" else False,
        "tpu_batchnorm_running_stats_hook_count": 20 if DEVICE.type == "tpu" else 0,
        "augmentation": "repository CXRDataset training augmentation enabled",
        "deviations": [
            "HF image bytes differ pixelwise from the preexisting staged copy; HF bytes are used for this run.",
            "The paper does not specify seed values; using the repo plan's reproducibility choice 42-51.",
            "The paper does not name the within-seed checkpoint epoch rule; using maximum validation AUROC.",
            "TorchTPU BatchNorm does not update running_mean/running_var in this installed build; a forward-pre-hook restores PyTorch EMA running-stat updates for validation.",
        ],
    }
    (RUN_ROOT / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    final: dict[str, dict] = {}
    for task in TASKS:
        task_slug = task.lower().replace(" ", "_")
        task_dir = RUN_ROOT / task_slug
        task_dir.mkdir(parents=True, exist_ok=True)
        train_ds = make_dataset("train", task, augmentation=True)
        val_ds = make_dataset("val", task, augmentation=False)
        test_ds = make_dataset("test", task, augmentation=False)
        print(json.dumps({"task": task, "split_examples": {"train": len(train_ds), "val": len(val_ds), "test": len(test_ds)}}), flush=True)
        results = []
        for seed in SEEDS:
            cached = load_completed_seed(task, seed, task_dir)
            if cached is not None:
                print(json.dumps({"task": task, "seed": seed, "status": "reusing completed checkpoint", "best_epoch": cached["best_epoch"], "best_val_auroc": cached["best_val_auroc"]}), flush=True)
                results.append(cached)
            else:
                results.append(train_seed(task, seed, train_ds, val_ds, task_dir))
        selected = sorted(results, key=lambda x: x["best_val_auroc"], reverse=True)[:5]
        tested = [evaluate_top_seed(task, item, test_ds, task_dir) for item in selected]
        test_aucs = [x["test_auroc"] for x in tested]
        final[task] = {
            "all_seeds": results,
            "selected_top_five": tested,
            "test_auroc_mean": float(np.mean(test_aucs)),
            "test_auroc_std_sample": float(np.std(test_aucs, ddof=1)) if len(test_aucs) > 1 else None,
        }
        (task_dir / "summary.json").write_text(json.dumps(final[task], indent=2) + "\n")
        print(json.dumps({"task": task, "selected": tested, "test_auroc_mean": final[task]["test_auroc_mean"], "test_auroc_std_sample": final[task]["test_auroc_std_sample"]}), flush=True)
    (RUN_ROOT / "summary.json").write_text(json.dumps(final, indent=2) + "\n")


if __name__ == "__main__":
    main()
