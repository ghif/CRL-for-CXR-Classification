#!/usr/bin/env python3
"""Evaluate TENT, CoTTA, EATA, and RoTTA TTA baselines across all CXR domains on TorchTPU."""

from __future__ import annotations

import argparse
import copy
import importlib.metadata
import json
import math
import random
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


DEFAULT_OUTPUT_ROOT = REPO / "chexpert/runs/tta_baselines_densenet121_cross_domain"
METHODS = ("tent", "cotta", "eata", "rotta")


def _select_bn_params(model: nn.Module) -> list[nn.Parameter]:
    params = []
    for name, param in model.named_parameters():
        if "bn" in name.lower() or "norm" in name.lower():
            param.requires_grad = True
            params.append(param)
        else:
            param.requires_grad = False
    return params


def _forward_probs(model: nn.Module, x: torch.Tensor, class_indices: list[int], device: torch.device) -> torch.Tensor:
    if device.type == "tpu":
        with torch.autocast("tpu", dtype=torch.bfloat16):
            logits = model(x)[:, class_indices]
    else:
        logits = model(x)[:, class_indices]
    return torch.sigmoid(logits.float())


@torch.inference_mode()
def _evaluate_logits(
    model: nn.Module, loader: DataLoader, class_indices: list[int], device: torch.device
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_y, all_scores = [], []
    for x, y in loader:
        x = x.to(device)
        if device.type == "tpu":
            with torch.autocast("tpu", dtype=torch.bfloat16):
                logits = model(x)[:, class_indices]
        else:
            logits = model(x)[:, class_indices]
        all_scores.append(logits.float().cpu().numpy())
        all_y.append(y.numpy())
    return np.concatenate(all_y), np.concatenate(all_scores)


def _binary_entropy(probs: torch.Tensor) -> torch.Tensor:
    p = probs.clamp(1e-7, 1.0 - 1e-7)
    return -p * torch.log(p) - (1.0 - p) * torch.log(1.0 - p)


def run_tent(
    model: nn.Module,
    loader: DataLoader,
    class_indices: list[int],
    device: torch.device,
    lr: float = 1e-3,
    steps: int = 1,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, dict]:
    seed_everything(seed)
    params = _select_bn_params(model)
    optimizer = torch.optim.Adam(params, lr=lr)
    t0 = time.perf_counter()
    model.train()
    for x, _ in loader:
        x = x.to(device)
        for _ in range(steps):
            probs = _forward_probs(model, x, class_indices, device)
            loss = _binary_entropy(probs).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    adapt_s = time.perf_counter() - t0
    y, scores = _evaluate_logits(model, loader, class_indices, device)
    return y, scores, {"adapt_seconds": round(adapt_s, 3)}


def run_cotta(
    model: nn.Module,
    loader: DataLoader,
    class_indices: list[int],
    device: torch.device,
    lr: float = 1e-3,
    ema_momentum: float = 0.999,
    restoration_prob: float = 0.01,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, dict]:
    seed_everything(seed)
    init_state = {k: v.detach().clone() for k, v in model.named_parameters()}
    teacher = copy.deepcopy(model).to(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    params = _select_bn_params(model)
    optimizer = torch.optim.Adam(params, lr=lr)
    t0 = time.perf_counter()
    for x, _ in loader:
        x = x.to(device)
        model.train()
        probs = _forward_probs(model, x, class_indices, device)
        loss = _binary_entropy(probs).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            for tp, sp in zip(teacher.parameters(), model.parameters()):
                tp.data.mul_(ema_momentum).add_(sp.data, alpha=1.0 - ema_momentum)
            if random.random() < restoration_prob:
                for name, param in model.named_parameters():
                    if name in init_state:
                        param.data.copy_(init_state[name])
    adapt_s = time.perf_counter() - t0
    y, scores = _evaluate_logits(teacher, loader, class_indices, device)
    return y, scores, {"adapt_seconds": round(adapt_s, 3)}


def run_eata(
    model: nn.Module,
    loader: DataLoader,
    class_indices: list[int],
    device: torch.device,
    lr: float = 1e-3,
    steps: int = 1,
    e_margin: float = 0.4,
    fisher_alpha: float = 50.0,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, dict]:
    seed_everything(seed)
    params = _select_bn_params(model)
    optimizer = torch.optim.Adam(params, lr=lr)
    anchor = {
        n: p.clone().detach()
        for n, p in model.named_parameters()
        if "bn" in n.lower() or "norm" in n.lower()
    }
    fishers = {
        n: torch.full_like(p, 1e-4, device=device)
        for n, p in model.named_parameters()
        if "bn" in n.lower() or "norm" in n.lower()
    }
    log2 = math.log(2.0)

    t0 = time.perf_counter()
    # Pass 0: Fisher estimation
    model.train()
    n_batches = 0
    for x, _ in loader:
        x = x.to(device)
        probs = _forward_probs(model, x, class_indices, device)
        ent = _binary_entropy(probs).mean()
        optimizer.zero_grad()
        ent.backward()
        with torch.no_grad():
            for n, p in model.named_parameters():
                if n in fishers and p.grad is not None:
                    fishers[n] += p.grad.pow(2)
        optimizer.zero_grad()
        n_batches += 1
    if n_batches > 0:
        for n in fishers:
            fishers[n] /= n_batches

    # Pass 1: reliable-sample entropy + Fisher regularization
    num_reliable = 0
    total_samples = 0
    for x, _ in loader:
        x = x.to(device)
        model.train()
        total_samples += x.size(0)
        for _ in range(steps):
            probs = _forward_probs(model, x, class_indices, device)
            ent = _binary_entropy(probs).mean(dim=1) / log2
            mask = ent < e_margin
            cnt = int(mask.sum().item())
            if cnt == 0:
                continue
            num_reliable += cnt
            ent_loss = ent[mask].mean()
            fisher_loss = torch.zeros((), device=device)
            for n, p in model.named_parameters():
                if n in fishers:
                    fisher_loss = fisher_loss + (fishers[n] * (p - anchor[n]).pow(2)).sum()
            loss = ent_loss + fisher_alpha * fisher_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    adapt_s = time.perf_counter() - t0
    y, scores = _evaluate_logits(model, loader, class_indices, device)
    return y, scores, {
        "adapt_seconds": round(adapt_s, 3),
        "reliable": num_reliable,
        "total_samples": total_samples,
    }


def run_rotta(
    model: nn.Module,
    loader: DataLoader,
    class_indices: list[int],
    device: torch.device,
    lr: float = 1e-3,
    steps: int = 1,
    ema_momentum: float = 0.999,
    memory_size: int = 64,
    nu: float = 0.001,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, dict]:
    seed_everything(seed)
    teacher = copy.deepcopy(model).to(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    global_bn = {}
    for name, module in model.named_modules():
        if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
            global_bn[name] = {
                "mean": module.running_mean.clone(),
                "var": module.running_var.clone(),
            }

    params = _select_bn_params(model)
    optimizer = torch.optim.Adam(params, lr=lr)
    mem_images: list[torch.Tensor] = []
    mem_ages: list[int] = []
    time_step = 0

    t0 = time.perf_counter()
    for x_cpu, _ in loader:
        for i in range(x_cpu.size(0)):
            if len(mem_images) < memory_size:
                mem_images.append(x_cpu[i].clone())
                mem_ages.append(time_step)
            else:
                oldest = mem_ages.index(min(mem_ages))
                mem_images[oldest] = x_cpu[i].clone()
                mem_ages[oldest] = time_step
        time_step += 1

        with torch.no_grad():
            for name, module in model.named_modules():
                if name in global_bn:
                    global_bn[name]["mean"] = (1.0 - nu) * global_bn[name]["mean"] + nu * module.running_mean
                    global_bn[name]["var"] = (1.0 - nu) * global_bn[name]["var"] + nu * module.running_var
                    module.running_mean.copy_(global_bn[name]["mean"])
                    module.running_var.copy_(global_bn[name]["var"])

        model.train()
        for _ in range(steps):
            n_samp = min(len(mem_images), 16)
            idxs = random.sample(range(len(mem_images)), n_samp)
            mem_x = torch.stack([mem_images[i] for i in idxs]).to(device)

            teacher.eval()
            with torch.no_grad():
                t_probs = _forward_probs(teacher, mem_x, class_indices, device).detach()
            s_probs = _forward_probs(model, mem_x, class_indices, device).clamp(1e-7, 1.0 - 1e-7)
            loss = (-t_probs * torch.log(s_probs) - (1.0 - t_probs) * torch.log(1.0 - s_probs)).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            for tp, sp in zip(teacher.parameters(), model.parameters()):
                tp.data.mul_(ema_momentum).add_(sp.data, alpha=1.0 - ema_momentum)
    adapt_s = time.perf_counter() - t0
    y, scores = _evaluate_logits(teacher, loader, class_indices, device)
    return y, scores, {"adapt_seconds": round(adapt_s, 3)}


RUNNERS = {
    "tent": run_tent,
    "cotta": run_cotta,
    "eata": run_eata,
    "rotta": run_rotta,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--methods", nargs="+", default=list(METHODS), choices=list(METHODS))
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
        "methods": args.methods,
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

            base_cpu = xrv.models.DenseNet(weights=WEIGHTS[source], cache_dir=str(args.weights_cache))
            base_cpu.op_threshs = None
            base_cpu.apply_sigmoid = False
            pathologies = list(base_cpu.pathologies)
            class_indices = [pathologies.index(head) for _, head in tasks]
            if device.type == "tpu":
                install_tpu_batchnorm_stats_fix(base_cpu)

            pair_entry = {
                "source": source,
                "target": target,
                "regime": regime,
                "examples": int(len(ds)),
                "tasks": [t for t, _ in tasks],
                "methods": {},
            }

            for method in args.methods:
                model = copy.deepcopy(base_cpu).to(device)
                y, scores, stats = RUNNERS[method](model, loader, class_indices, device, seed=args.seed)
                m = compute_metrics(y, scores, tasks)
                pair_entry["methods"][method] = {"metrics": m, "stats": stats}

                for task_name, _ in tasks:
                    csv_rows.append(
                        {
                            "pair": pair_key,
                            "source": source,
                            "target": target,
                            "regime": regime,
                            "method": method,
                            "task": task_name,
                            "examples": int(len(ds)),
                            "auroc": round(m[task_name]["auroc"], 6),
                            "auprc": round(m[task_name]["auprc"], 6),
                        }
                    )

                print(
                    json.dumps(
                        {
                            "pair": pair_key,
                            "method": method,
                            "macro_auroc": round(m["_macro"]["macro_auroc"], 4),
                            "macro_auprc": round(m["_macro"]["macro_auprc"], 4),
                            "adapt_s": stats["adapt_seconds"],
                        }
                    ),
                    flush=True,
                )
                del model

            summary["evaluations"][pair_key] = pair_entry
            (args.output_root / "summary.partial.json").write_text(json.dumps(summary, indent=2) + "\n")
            pd.DataFrame(csv_rows).to_csv(args.output_root / "comparison.partial.csv", index=False)
            del base_cpu

        del images, labels, ds, loader

    (args.output_root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    pd.DataFrame(csv_rows).to_csv(args.output_root / "comparison.csv", index=False)
    print(json.dumps({"status": "done", "output_root": str(args.output_root)}), flush=True)


if __name__ == "__main__":
    main()
