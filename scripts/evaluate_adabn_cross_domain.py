#!/usr/bin/env python3
"""Evaluate AdaBN (Li et al., ICLR 2017 Workshop) across all CXR domains on TorchTPU."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torchxrayvision as xrv


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluate_cowa_cross_domain import (  # noqa: E402
    ALL_TARGETS,
    DEFAULT_WEIGHTS_CACHE,
    WEIGHTS,
    compute_metrics,
    install_tpu_batchnorm_stats_fix,
    load_target_tensors,
    seed_everything,
)


DEFAULT_OUTPUT_ROOT = REPO / "chexpert/runs/adabn_densenet121_cross_domain"


class AdaBNTorchTPU:
    """Adaptive Batch Normalization (cumulative moving average, momentum=None) for TorchTPU."""

    def __init__(self, model: nn.Module, class_indices: list[int], device: torch.device, seed: int = 42):
        self.model = model
        self.class_indices = class_indices
        self.device = device
        self.seed = seed

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
        bn_layers = 0
        for module in self.model.modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                module.reset_running_stats()
                module.momentum = None
                bn_layers += 1

        t0 = time.perf_counter()
        self.model.train()
        with torch.no_grad():
            for x, _ in loader:
                x = x.to(self.device)
                if self.device.type == "tpu":
                    with torch.autocast("tpu", dtype=torch.bfloat16):
                        _ = self.model(x)
                else:
                    _ = self.model(x)
        adapt_seconds = time.perf_counter() - t0

        y, scores = self.evaluate(loader)
        return y, scores, {"bn_layers": bn_layers, "adapt_seconds": round(adapt_seconds, 3)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", nargs="+", default=list(WEIGHTS.keys()), choices=list(WEIGHTS.keys()))
    parser.add_argument("--targets", nargs="+", default=ALL_TARGETS, choices=ALL_TARGETS)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--weights-cache", type=Path, default=DEFAULT_WEIGHTS_CACHE)
    parser.add_argument("--device", default="tpu")
    parser.add_argument("--batch-size", type=int, default=64)
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
        "method": "AdaBN (cumulative momentum=None)",
        "batch_size": args.batch_size,
        "seed": args.seed,
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

            adapter = AdaBNTorchTPU(base_model, class_indices=class_indices, device=device, seed=args.seed)
            y_base, scores_base = adapter.evaluate(loader)
            m_base = compute_metrics(y_base, scores_base, tasks)

            y_ada, scores_ada, stats = adapter.adapt_and_evaluate(loader)
            m_ada = compute_metrics(y_ada, scores_ada, tasks)

            summary["evaluations"][pair_key] = {
                "source": source,
                "target": target,
                "regime": regime,
                "examples": int(len(ds)),
                "tasks": [t for t, _ in tasks],
                "no_adapt": m_base,
                "adabn": m_ada,
                "delta_macro_auroc": round(m_ada["_macro"]["macro_auroc"] - m_base["_macro"]["macro_auroc"], 6),
                "delta_macro_auprc": round(m_ada["_macro"]["macro_auprc"] - m_base["_macro"]["macro_auprc"], 6),
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
                        "adabn_auroc": round(m_ada[task_name]["auroc"], 6),
                        "delta_auroc": round(m_ada[task_name]["auroc"] - m_base[task_name]["auroc"], 6),
                        "no_adapt_auprc": round(m_base[task_name]["auprc"], 6),
                        "adabn_auprc": round(m_ada[task_name]["auprc"], 6),
                        "delta_auprc": round(m_ada[task_name]["auprc"] - m_base[task_name]["auprc"], 6),
                    }
                )

            print(
                json.dumps(
                    {
                        "pair": pair_key,
                        "no_adapt_auroc": round(m_base["_macro"]["macro_auroc"], 4),
                        "adabn_auroc": round(m_ada["_macro"]["macro_auroc"], 4),
                        "delta": round(m_ada["_macro"]["macro_auroc"] - m_base["_macro"]["macro_auroc"], 4),
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
