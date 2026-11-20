#!/usr/bin/env python3
"""Evaluate CoWA (MICCAI 2026) Test-Time Adaptation vs. No Adapt. across all CXR domains on TorchTPU."""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import importlib.metadata
import json
import random
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
import torch.nn as nn
from skimage.io import imread
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset
import torchxrayvision as xrv


REPO = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = REPO / "chexpert/runs/cowa_densenet121_cross_domain"
DEFAULT_WEIGHTS_CACHE = REPO / "chexpert/runs/xrv_densenet121_chexpert_padchest/weights_cache"

WEIGHTS = {
    "chexpert": "densenet121-res224-chex",
    "padchest": "densenet121-res224-pc",
    "mimic_ch": "densenet121-res224-mimic_ch",
    "nih": "densenet121-res224-nih",
}

TASKS_6 = [
    ("Atelectasis", "Atelectasis"),
    ("Cardiomegaly", "Cardiomegaly"),
    ("Pleural Effusion", "Effusion"),
    ("Consolidation", "Consolidation"),
    ("Pneumothorax", "Pneumothorax"),
    ("Edema", "Edema"),
]
TASKS_5 = TASKS_6[:5]

ALL_TARGETS = ["chexpert", "padchest", "cxr_rait", "mimic_cxr", "xraydar", "nih"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def install_tpu_batchnorm_stats_fix(model: nn.Module) -> int:
    """Restore PyTorch BatchNorm EMA buffers on TorchTPU builds that omit updates in train()."""
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


# ---------------------------------------------------------------------------
# Target Dataset Loaders -> (images: Tensor[N,1,224,224], targets: Tensor[N,C], tasks)
# ---------------------------------------------------------------------------

def _load_images_parallel(paths: list[str], resize_to_224: bool, num_workers: int = 8) -> torch.Tensor:
    from concurrent.futures import ThreadPoolExecutor

    resizer = xrv.datasets.XRayResizer(224, engine="skimage") if resize_to_224 else None

    def _read_one(path: str) -> np.ndarray:
        img = imread(path)
        if img.ndim == 3:
            img = img[..., 0]
        maxval = 65535.0 if img.dtype == np.uint16 else 255.0
        arr = img.astype(np.float32, copy=False)[None, :, :]
        arr = xrv.datasets.normalize(arr, maxval=maxval)
        if resizer is not None:
            arr = resizer(arr)
        return arr

    with ThreadPoolExecutor(max_workers=num_workers) as ex:
        arrays = list(ex.map(_read_one, paths))
    return torch.from_numpy(np.stack(arrays, axis=0))


def load_target_tensors(target: str, num_workers: int = 8) -> tuple[torch.Tensor, torch.Tensor, list[tuple[str, str]]]:
    if target == "chexpert":
        csv_path = REPO / "datafiles/chexpert/chexpert.sample.test.csv"
        hf_dir = Path("/mnt/data/dataset/chexpert/hf/data")
        img_dir = Path("/mnt/data/dataset/chexpert/hf_preprocessed")
        df = pd.read_csv(csv_path)

        def _norm(p: str) -> str:
            name = re.sub(r"^preproc_224x224/", "", p).replace(".jpg.png", ".jpg")
            m = re.fullmatch(r"(patient\d+)_(study\d+)_(view\d+_.*\.jpg)", name)
            return f"CheXpert-v1.0-small/train/{m.group(1)}/{m.group(2)}/{m.group(3)}"

        df["hf_path"] = df["path_preproc"].astype(str).map(_norm)
        wanted = set(df["hf_path"])
        task_cols = [t for t, _ in TASKS_6]
        hf_parts = []
        for shard in sorted(hf_dir.glob("train-*.parquet")):
            part = pq.read_table(shard, columns=["Path", *task_cols]).to_pandas()
            part = part[part["Path"].isin(wanted)]
            if not part.empty:
                hf_parts.append(part)
        hf_df = pd.concat(hf_parts, ignore_index=True).drop_duplicates(subset=["Path"])
        merged = df.merge(hf_df, left_on="hf_path", right_on="Path", how="inner", suffixes=("_csv", ""))
        paths = [str(img_dir / p) for p in merged["path_preproc"].astype(str)]
        y = np.stack([(merged[t] == 3).to_numpy(dtype=np.int64) for t in task_cols], axis=1)
        return _load_images_parallel(paths, resize_to_224=False, num_workers=num_workers), torch.from_numpy(y), TASKS_6

    if target == "padchest":
        stage_dir = Path("/mnt/data/dataset/padchest/input-stage/images-224")
        manifest = stage_dir / "padchest-stage-manifest.jsonl"
        pc_csv = Path("/mnt/data/dataset/padchest/metadata/PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv")
        m_rows = []
        with manifest.open("r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                if rec.get("kind") == "item" and rec.get("split") == "test":
                    m_rows.append((rec["image_id"], rec["relative_path"]))
        mdf = pd.DataFrame(m_rows, columns=["ImageID", "relative_path"]).drop_duplicates("ImageID")
        pc_df = pd.read_csv(pc_csv, usecols=["ImageID", "Projection", "Labels"], low_memory=False)
        merged = mdf.merge(pc_df, on="ImageID", how="inner")
        merged = merged[merged["Projection"].isin({"PA", "AP", "AP_horizontal"})].copy().reset_index(drop=True)
        lbl_map = {
            "Atelectasis": {"atelectasis", "atelectasis basal", "laminar atelectasis", "lobar atelectasis", "round atelectasis", "segmental atelectasis", "total atelectasis"},
            "Cardiomegaly": {"cardiomegaly"},
            "Pleural Effusion": {"pleural effusion", "loculated pleural effusion", "loculated fissural effusion"},
            "Consolidation": {"consolidation"},
            "Pneumothorax": {"pneumothorax", "hydropneumothorax"},
            "Edema": {"pulmonary edema"},
        }
        parsed = []
        for raw in merged["Labels"].fillna("[]"):
            try:
                parsed.append({str(x).strip().lower() for x in ast.literal_eval(raw)})
            except (ValueError, SyntaxError):
                parsed.append(set())
        y = np.stack([[1 if (lbls & lbl_map[t]) else 0 for lbls in parsed] for t, _ in TASKS_6], axis=1).astype(np.int64)
        paths = [str(stage_dir / p) for p in merged["relative_path"]]
        return _load_images_parallel(paths, resize_to_224=False, num_workers=num_workers), torch.from_numpy(y), TASKS_6

    if target == "cxr_rait":
        df = pd.read_csv("/mnt/data/dataset/cxr_rait/labels_5pathologies.csv")
        img_root = Path("/mnt/data/dataset/cxr_rait/preprocessed_224")
        paths = [str(img_root / f"{Path(fn).stem}.png") for fn in df["cv_filename"]]
        cols = ["atelectasis", "cardiomegaly", "pleural_effusion", "consolidation", "pneumothorax"]
        y = df[cols].to_numpy(dtype=np.int64)
        return _load_images_parallel(paths, resize_to_224=False, num_workers=num_workers), torch.from_numpy(y), TASKS_5

    if target == "mimic_cxr":
        df = pd.read_csv("/mnt/data/dataset/mimic_cxr/validate_frontal_6pathologies.csv")
        img_root = Path("/mnt/data/dataset/mimic_cxr/official_data_iccv_final")
        paths = [str(img_root / p) for p in df["relative_path"]]
        y = df[[t for t, _ in TASKS_6]].to_numpy(dtype=np.int64)
        return _load_images_parallel(paths, resize_to_224=True, num_workers=num_workers), torch.from_numpy(y), TASKS_6

    if target == "xraydar":
        root = Path("/mnt/data/dataset/xraydar")
        rows = []
        paths = []
        keys = ["atelectasis", "cardiomegaly", "pleural_effusion", "consolidation", "pneumothorax"]
        with (root / "annotations.jsonl").open("r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                lbls = set(rec.get("consensus_labels", []))
                paths.append(str(root / rec["image_file"]))
                rows.append([int(k in lbls) for k in keys])
        y = np.asarray(rows, dtype=np.int64)
        return _load_images_parallel(paths, resize_to_224=True, num_workers=num_workers), torch.from_numpy(y), TASKS_5

    if target == "nih":
        root = Path("/mnt/data/dataset/nih_cxr")
        test_set = {l.strip() for l in (root / "test_list.txt").read_text().splitlines() if l.strip()}
        df = pd.read_csv(root / "Data_Entry_2017.csv")
        df = df[df["Image Index"].isin(test_set)].copy().reset_index(drop=True)
        paths = [str(root / "test_images" / n) for n in df["Image Index"]]
        fsets = df["Finding Labels"].fillna("").map(lambda s: {x.strip() for x in s.split("|") if x.strip()})
        nih_names = ["Atelectasis", "Cardiomegaly", "Effusion", "Consolidation", "Pneumothorax", "Edema"]
        y = np.stack([[1 if nm in fs else 0 for fs in fsets] for nm in nih_names], axis=1).astype(np.int64)
        return _load_images_parallel(paths, resize_to_224=True, num_workers=num_workers), torch.from_numpy(y), TASKS_6

    raise ValueError(f"Unknown target: {target}")


# ---------------------------------------------------------------------------
# CoWA (TorchTPU-compatible port of woojin716/CoWA/src/cowa.py)
# ---------------------------------------------------------------------------

class CoWATorchTPU:
    """Co-occurrence Weighted Adaptation (Jeong et al., MICCAI 2026) ported to TorchTPU."""

    def __init__(
        self,
        model: nn.Module,
        class_indices: list[int],
        device: torch.device,
        lr: float = 1e-3,
        steps: int = 1,
        cooccur_threshold: float = 0.5,
        pattern_temp: float = 0.1,
        min_weight: float = 0.01,
        seed: int = 42,
    ):
        self.model = model
        self.class_indices = class_indices
        self.device = device
        self.lr = lr
        self.steps = steps
        self.cooccur_threshold = cooccur_threshold
        self.pattern_temp = pattern_temp
        self.min_weight = min_weight
        self.seed = seed

        self.params = []
        for name, param in model.named_parameters():
            if "bn" in name.lower() or "norm" in name.lower():
                param.requires_grad = True
                self.params.append(param)
            else:
                param.requires_grad = False

        self.optimizer = torch.optim.Adam(self.params, lr=self.lr)
        self._cooccur_sum: torch.Tensor | None = None
        self._sample_count = 0
        self.cooccur_matrix: torch.Tensor | None = None

    def _update_cooccurrence(self, probs: torch.Tensor) -> None:
        binary = (probs > self.cooccur_threshold).float()
        batch_cooccur = torch.matmul(binary.t(), binary)
        if self._cooccur_sum is None:
            self._cooccur_sum = batch_cooccur
        else:
            self._cooccur_sum = self._cooccur_sum + batch_cooccur
        self._sample_count += probs.size(0)

        cooccur = self._cooccur_sum / self._sample_count
        diag = torch.diag(cooccur)
        diag_sqrt = torch.sqrt(diag.unsqueeze(0) * diag.unsqueeze(1))
        cooccur = cooccur / (diag_sqrt + 1e-8)
        cooccur.fill_diagonal_(1.0)
        self.cooccur_matrix = cooccur

    def _compute_pattern_consistency(self, probs: torch.Tensor) -> torch.Tensor:
        C = probs.size(1)
        pred_cooccur = probs.unsqueeze(2) * probs.unsqueeze(1)
        expected = self.cooccur_matrix.unsqueeze(0)
        mask = torch.triu(torch.ones(C, C, device=probs.device), diagonal=1)
        diff_sq = ((pred_cooccur - expected) ** 2) * mask.unsqueeze(0)
        distances = diff_sq.sum(dim=(1, 2)) / (mask.sum() + 1e-8)
        return torch.exp(-distances / self.pattern_temp)

    @staticmethod
    def _entropy_per_sample(probs: torch.Tensor) -> torch.Tensor:
        probs_c = probs.clamp(1e-7, 1.0 - 1e-7)
        entropy = -probs_c * torch.log(probs_c) - (1.0 - probs_c) * torch.log(1.0 - probs_c)
        return entropy.mean(dim=1)

    def _forward_probs(self, x: torch.Tensor) -> torch.Tensor:
        if self.device.type == "tpu":
            with torch.autocast("tpu", dtype=torch.bfloat16):
                logits = self.model(x)[:, self.class_indices]
        else:
            logits = self.model(x)[:, self.class_indices]
        return torch.sigmoid(logits.float())

    @torch.inference_mode()
    def evaluate(self, loader: DataLoader) -> tuple[np.ndarray, np.ndarray]:
        self.model.eval()
        all_y, all_scores = [], []
        for x, y in loader:
            x = x.to(self.device)
            if self.device.type == "tpu":
                with torch.autocast("tpu", dtype=torch.bfloat16):
                    logits = self.model(x)[:, self.class_indices]
            else:
                logits = self.model(x)[:, self.class_indices]
            all_scores.append(logits.float().cpu().numpy())
            all_y.append(y.numpy())
        return np.concatenate(all_y), np.concatenate(all_scores)

    def adapt_and_evaluate(self, loader: DataLoader) -> tuple[np.ndarray, np.ndarray, dict]:
        seed_everything(self.seed)
        self._cooccur_sum = None
        self._sample_count = 0
        self.cooccur_matrix = None

        total_loss = 0.0
        total_w = 0.0
        total_cons = 0.0
        n_batches = 0

        t0 = time.perf_counter()
        for x, _ in loader:
            x = x.to(self.device)
            # 1) Update target co-occurrence matrix in eval mode
            self.model.eval()
            with torch.no_grad():
                probs_eval = self._forward_probs(x)
                self._update_cooccurrence(probs_eval.cpu())

            # 2) Weighted entropy adaptation on BN affine parameters
            cooccur_dev = self.cooccur_matrix.to(self.device)
            self.cooccur_matrix = cooccur_dev
            self.model.train()
            for _ in range(self.steps):
                probs = self._forward_probs(x)
                entropy = self._entropy_per_sample(probs)
                consistency = self._compute_pattern_consistency(probs)
                weights = torch.clamp(consistency, min=self.min_weight)
                loss = (entropy * weights).mean()

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            total_loss += float(loss.detach().cpu())
            total_w += float(weights.mean().detach().cpu())
            total_cons += float(consistency.mean().detach().cpu())
            n_batches += 1
            self.cooccur_matrix = self.cooccur_matrix.cpu()

        adapt_time = time.perf_counter() - t0
        y, scores = self.evaluate(loader)
        stats = {
            "adapt_seconds": round(adapt_time, 3),
            "avg_weighted_entropy": round(total_loss / max(n_batches, 1), 6),
            "avg_sample_weight": round(total_w / max(n_batches, 1), 4),
            "avg_consistency": round(total_cons / max(n_batches, 1), 4),
        }
        return y, scores, stats


def compute_metrics(y: np.ndarray, scores: np.ndarray, tasks: list[tuple[str, str]]) -> dict:
    out = {}
    aurocs, auprcs = [], []
    for idx, (task_name, _) in enumerate(tasks):
        yt = y[:, idx]
        ys = scores[:, idx]
        auroc = float(roc_auc_score(yt, ys))
        auprc = float(average_precision_score(yt, ys))
        aurocs.append(auroc)
        auprcs.append(auprc)
        out[task_name] = {"auroc": auroc, "auprc": auprc}
    out["_macro"] = {"macro_auroc": float(np.mean(aurocs)), "macro_auprc": float(np.mean(auprcs))}
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", nargs="+", default=list(WEIGHTS.keys()), choices=list(WEIGHTS.keys()))
    parser.add_argument("--targets", nargs="+", default=ALL_TARGETS, choices=ALL_TARGETS)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--weights-cache", type=Path, default=DEFAULT_WEIGHTS_CACHE)
    parser.add_argument("--device", default="tpu")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--cooccur-threshold", type=float, default=0.5)
    parser.add_argument("--pattern-temp", type=float, default=0.1)
    parser.add_argument("--min-weight", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == "tpu":
        import torch_tpu  # noqa: F401

        if not torch.accelerator.is_available() or torch.accelerator.current_accelerator().type != "tpu":
            raise RuntimeError("--device tpu requested, but TorchTPU reports no available TPU")
    torch.set_num_threads(32 if device.type == "tpu" else 4)

    args.output_root.mkdir(parents=True, exist_ok=True)
    args.weights_cache.mkdir(parents=True, exist_ok=True)

    summary = {
        "device": str(device),
        "hyperparams": {
            "batch_size": args.batch_size,
            "lr": args.lr,
            "steps": args.steps,
            "cooccur_threshold": args.cooccur_threshold,
            "pattern_temp": args.pattern_temp,
            "min_weight": args.min_weight,
            "seed": args.seed,
        },
        "torchxrayvision_version": importlib.metadata.version("torchxrayvision"),
        "evaluations": {},
    }
    csv_rows = []

    for target in args.targets:
        print(json.dumps({"status": "loading_target", "target": target}), flush=True)
        images, labels, tasks = load_target_tensors(target)
        ds = TensorDataset(images, labels)
        g = torch.Generator().manual_seed(args.seed)
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, generator=g)

        for source in args.sources:
            pair_key = f"{source}_to_{target}"
            regime = "in_domain" if (source == target or (source == "mimic_ch" and target == "mimic_cxr")) else "cross_domain"

            base_model = xrv.models.DenseNet(weights=WEIGHTS[source], cache_dir=str(args.weights_cache))
            base_model.op_threshs = None
            base_model.apply_sigmoid = False
            pathologies = list(base_model.pathologies)
            class_indices = [pathologies.index(head) for _, head in tasks]
            if device.type == "tpu":
                install_tpu_batchnorm_stats_fix(base_model)
            base_model = base_model.to(device)

            adapter = CoWATorchTPU(
                model=base_model,
                class_indices=class_indices,
                device=device,
                lr=args.lr,
                steps=args.steps,
                cooccur_threshold=args.cooccur_threshold,
                pattern_temp=args.pattern_temp,
                min_weight=args.min_weight,
                seed=args.seed,
            )

            # 1) Unadapted baseline (No Adapt.)
            y_base, scores_base = adapter.evaluate(loader)
            m_base = compute_metrics(y_base, scores_base, tasks)

            # 2) CoWA TTA
            y_cowa, scores_cowa, stats = adapter.adapt_and_evaluate(loader)
            m_cowa = compute_metrics(y_cowa, scores_cowa, tasks)

            summary["evaluations"][pair_key] = {
                "source": source,
                "target": target,
                "regime": regime,
                "examples": int(len(ds)),
                "tasks": [t for t, _ in tasks],
                "no_adapt": m_base,
                "cowa": m_cowa,
                "delta_macro_auroc": round(m_cowa["_macro"]["macro_auroc"] - m_base["_macro"]["macro_auroc"], 6),
                "delta_macro_auprc": round(m_cowa["_macro"]["macro_auprc"] - m_base["_macro"]["macro_auprc"], 6),
                "stats": stats,
            }

            for task_name, _ in tasks:
                csv_rows.append(
                    {
                        "pair": pair_key,
                        "source": source,
                        "target": target,
                        "regime": regime,
                        "task": task_name,
                        "examples": int(len(ds)),
                        "no_adapt_auroc": round(m_base[task_name]["auroc"], 6),
                        "cowa_auroc": round(m_cowa[task_name]["auroc"], 6),
                        "delta_auroc": round(m_cowa[task_name]["auroc"] - m_base[task_name]["auroc"], 6),
                        "no_adapt_auprc": round(m_base[task_name]["auprc"], 6),
                        "cowa_auprc": round(m_cowa[task_name]["auprc"], 6),
                        "delta_auprc": round(m_cowa[task_name]["auprc"] - m_base[task_name]["auprc"], 6),
                    }
                )

            print(
                json.dumps(
                    {
                        "pair": pair_key,
                        "no_adapt_auroc": round(m_base["_macro"]["macro_auroc"], 4),
                        "cowa_auroc": round(m_cowa["_macro"]["macro_auroc"], 4),
                        "delta": round(m_cowa["_macro"]["macro_auroc"] - m_base["_macro"]["macro_auroc"], 4),
                        "adapt_s": stats["adapt_seconds"],
                    }
                ),
                flush=True,
            )

        del images, labels, ds, loader

    (args.output_root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    pd.DataFrame(csv_rows).to_csv(args.output_root / "comparison.csv", index=False)
    print(json.dumps({"status": "done", "output_root": str(args.output_root)}), flush=True)


if __name__ == "__main__":
    main()
