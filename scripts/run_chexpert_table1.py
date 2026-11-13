#!/usr/bin/env python3
"""Run the 24 CheXpert-only Table 1 configurations and select five seeds."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
import traceback
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import numpy as np
import torch

from prediction.chexpert_table1 import (
    METHODS,
    MODEL_TYPES,
    TASKS,
    build_datasets,
    evaluate_checkpoint,
    train_one,
)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def _configure_external_runtime(runtime_root: str | Path) -> None:
    root = Path(runtime_root).resolve()
    if not root.is_relative_to(Path("/mnt/data")):
        raise RuntimeError("Runtime caches and temporary files must stay on /mnt/data")
    cache_root = root / "torch-cache"
    temp_root = root / "tmp"
    cache_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    temp_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.environ["TORCH_HOME"] = str(cache_root)
    os.environ["XDG_CACHE_HOME"] = str(cache_root)
    os.environ["TMPDIR"] = str(temp_root)
    os.environ["TMP"] = str(temp_root)
    os.environ["TEMP"] = str(temp_root)
    torch.hub.set_dir(str(cache_root / "hub"))


def _runtime_provenance() -> dict:
    repo_root = Path(__file__).resolve().parents[1]
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo_root, capture_output=True, text=True
    )
    dirty = subprocess.run(
        ["git", "status", "--porcelain"], cwd=repo_root, capture_output=True, text=True
    )
    code_paths = (
        "prediction/backbones.py",
        "prediction/datasets/cxr.py",
        "prediction/chexpert_table1.py",
        "scripts/run_chexpert_table1.py",
    )
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": str(torch.__version__),
        "torch_tpu": _version("torch_tpu"),
        "libtpu": _version("libtpu"),
        "pytorch_lightning": _version("pytorch-lightning"),
        "torch_xla": _version("torch-xla"),
        "device": "torch_tpu device='tpu'",
        "torch_home": os.environ.get("TORCH_HOME"),
        "tmpdir": os.environ.get("TMPDIR"),
        "git_head": revision.stdout.strip() if revision.returncode == 0 else None,
        "working_tree_dirty": bool(dirty.stdout.strip()),
        "source_sha256": {path: _file_sha256(repo_root / path) for path in code_paths},
    }


def _logged_call(log_path: Path, function, *args, **kwargs):
    log_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with log_path.open("a") as log:
        try:
            with redirect_stdout(log), redirect_stderr(log):
                return function(*args, **kwargs)
        except BaseException:
            traceback.print_exc(file=log)
            raise


def run_smoke(args: argparse.Namespace) -> None:
    if not args.confirm_original_backbone:
        raise RuntimeError(
            "Smoke validation is held until the requested original-backbone protocol is clarified"
        )
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    root.chmod(0o700)
    task, model_name, method = args.smoke_task, args.smoke_model, args.smoke_method
    datasets = build_datasets(args.data_root, task, method, nsamples=2)
    run_dir = Path(args.output_dir) / task.lower().replace(" ", "_") / model_name / method / f"seed-{args.seed_start}"
    result = _logged_call(
        run_dir / "training.log",
        train_one,
        data_root=args.data_root,
        output_dir=args.output_dir,
        task=task,
        model_name=model_name,
        method=method,
        seed=args.seed_start,
        epochs=20,
        batch_size=32,
        nsamples=2,
        num_workers=args.workers,
        device_name="tpu",
        max_steps=5,
        datasets=datasets,
        runtime_provenance=_runtime_provenance(),
    )
    test_metrics = _logged_call(
        run_dir / "test-evaluation.log",
        evaluate_checkpoint,
        result["checkpoint"],
        data_root=args.data_root,
        task=task,
        model_name=model_name,
        method=method,
        batch_size=32,
        nsamples=2,
        num_workers=args.workers,
        device_name="tpu",
        datasets=datasets,
        predictions_path=run_dir / "test-predictions.csv",
    )
    outcome = {"smoke": result, "test": test_metrics, "bounded_steps": 5}
    path = root / "smoke-result.json"
    path.write_text(json.dumps(outcome, indent=2) + "\n")
    os.chmod(path, 0o600)
    print(json.dumps({"smoke_steps": result["steps"], "val_auroc": result["best_val_auroc"], "test_auroc": test_metrics["auroc"]}), flush=True)


def run_full(args: argparse.Namespace) -> None:
    if not args.confirm_original_backbone:
        raise RuntimeError(
            "Full training is held until the requested original-backbone validation protocol is clarified"
        )
    if not args.confirm_model_preflights:
        raise RuntimeError("All required backbone/group-sampling smoke preflights must pass first")
    data_root = Path(args.data_root)
    manifest_path = data_root / "staging_manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError("Full training requires a completed selective-staging manifest")
    manifest = json.loads(manifest_path.read_text())
    if (
        not manifest.get("complete")
        or manifest.get("scope") != "full_table1"
        or manifest.get("selected_images") != 71096
        or manifest.get("missing_images") != 0
        or manifest.get("zip_entry_crc32_verified_images") != 71096
        or manifest.get("verified_png_count") != 71096
    ):
        raise RuntimeError("Full staging manifest does not prove the complete verified image subset")
    image_dir = data_root / "preproc_224x224"
    if len(list(image_dir.glob("*.png"))) != 71096:
        raise RuntimeError("Staged image count changed since integrity verification")
    for split_name in ("train", "val", "test"):
        if not (data_root / "datafiles" / "chexpert" / f"chexpert.sample.{split_name}.csv").is_file():
            raise RuntimeError(f"Required {split_name} manifest is missing")
    if shutil.disk_usage(data_root).free < 20 * 1024**3:
        raise RuntimeError("Less than 20 GiB remains free for checkpoints and logs")
    # Fail before constructing the configuration matrix if native TorchTPU is unavailable.
    torch.empty(1, device="tpu")

    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    root.chmod(0o700)
    summary_rows = []
    all_selected = []
    seeds = list(range(args.seed_start, args.seed_start + args.seeds))
    provenance = {
        **_runtime_provenance(),
        "table": "Table 1 CheXpert column",
        "tasks": list(TASKS),
        "backbones": list(MODEL_TYPES),
        "methods": list(METHODS),
        "seed_values": seeds,
        "epochs": args.epochs,
        "batch_size": 32,
        "batch_size_semantics": {
            "baseline": "32 images per optimizer step",
            "invariant": "32 groups x P=2 = 64 image inputs per optimizer step",
        },
        "invariant_samples_per_iteration": 2,
        "invariant_objective_gap": "repository mean-squared distance minus embedding entropy; paper prose/pseudocode sign and distance reduction are ambiguous",
        "lr_backbone": 0.001,
        "lr_classifier": 0.0001,
        "optimizer": "Adam",
        "image_size": [224, 224],
        "device": "torch_tpu (tpu)",
        "checkpoint_selection": "maximum validation AUROC within each seed, then top five seeds per configuration",
        "validation_bce_sentence_conflict": "Paper implementation details say largest validation BCE; Section 3.2 specifies top five by validation AUROC. This run follows Section 3.2 for seed ranking.",
        "paper_does_not_report_seed_values": True,
        "staging_manifest_sha256": _file_sha256(Path(args.data_root) / "staging_manifest.json"),
        "preflight_report_sha256": _file_sha256(Path(args.data_root) / "preflight.json"),
    }
    (root / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")

    for task in TASKS:
        for model_name in MODEL_TYPES:
            for method in METHODS:
                print(f"START task={task}; model={model_name}; method={method}", flush=True)
                datasets = build_datasets(args.data_root, task, method, nsamples=2)
                candidates = []
                for seed in seeds:
                    run_dir = root / task.lower().replace(" ", "_") / model_name / method / f"seed-{seed}"
                    metrics_path = run_dir / "metrics.json"
                    checkpoint_path = run_dir / "best.ckpt"
                    if metrics_path.exists() and checkpoint_path.exists():
                        previous = json.loads(metrics_path.read_text())
                        if previous.get("epochs_completed") == args.epochs:
                            result = previous
                        else:
                            result = None
                    else:
                        result = None
                    if result is None:
                        failure_path = run_dir / "failure-attempt-1.json"
                        try:
                            result = _logged_call(
                                run_dir / "training.log",
                                train_one,
                                data_root=args.data_root,
                                output_dir=args.output_dir,
                                task=task,
                                model_name=model_name,
                                method=method,
                                seed=seed,
                                epochs=args.epochs,
                                batch_size=32,
                                nsamples=2,
                                num_workers=args.workers,
                                device_name="tpu",
                                datasets=datasets,
                                runtime_provenance=provenance,
                            )
                        except Exception as first_error:
                            failure_path.write_text(json.dumps({
                                "attempt": 1, "seed": seed,
                                "error_type": type(first_error).__name__,
                                "message": str(first_error),
                            }, indent=2) + "\n")
                            print(f"RETRY task={task}; model={model_name}; method={method}; seed={seed}", flush=True)
                            try:
                                result = _logged_call(
                                    run_dir / "training.log",
                                    train_one,
                                    data_root=args.data_root,
                                    output_dir=args.output_dir,
                                    task=task,
                                    model_name=model_name,
                                    method=method,
                                    seed=seed,
                                    epochs=args.epochs,
                                    batch_size=32,
                                    nsamples=2,
                                    num_workers=args.workers,
                                    device_name="tpu",
                                    datasets=datasets,
                                    runtime_provenance=provenance,
                                )
                            except Exception as second_error:
                                (run_dir / "failure-attempt-2.json").write_text(json.dumps({
                                    "attempt": 2, "seed": seed,
                                    "error_type": type(second_error).__name__,
                                    "message": str(second_error),
                                }, indent=2) + "\n")
                                raise
                    candidates.append(result)
                    print(
                        f"SEED task={task}; model={model_name}; method={method}; seed={seed}; "
                        f"best_val_auroc={result['best_val_auroc']:.6f}",
                        flush=True,
                    )
                    gc.collect()

                validation_by_seed = sorted(candidates, key=lambda row: int(row["seed"]))
                candidates.sort(
                    key=lambda row: (float(row["best_val_auroc"]), -int(row["seed"])),
                    reverse=True,
                )
                selected = candidates[:5]
                test_results = []
                for candidate in selected:
                    seed_run_dir = Path(candidate["checkpoint"]).parent
                    eval_path = seed_run_dir / "test-evaluation.json"
                    prediction_path = seed_run_dir / "test-predictions.csv"
                    checkpoint_hash = candidate.get("checkpoint_sha256") or _file_sha256(
                        Path(candidate["checkpoint"])
                    )
                    if (
                        eval_path.is_file()
                        and prediction_path.is_file()
                        and checkpoint_hash
                        and json.loads(eval_path.read_text()).get("checkpoint_sha256") == checkpoint_hash
                    ):
                        metrics = json.loads(eval_path.read_text())
                    else:
                        try:
                            metrics = _logged_call(
                                seed_run_dir / "test-evaluation.log",
                                evaluate_checkpoint,
                                candidate["checkpoint"],
                                data_root=args.data_root,
                                task=task,
                                model_name=model_name,
                                method=method,
                                batch_size=32,
                                nsamples=2,
                                num_workers=args.workers,
                                device_name="tpu",
                                datasets=datasets,
                                predictions_path=prediction_path,
                            )
                        except Exception as first_error:
                            (seed_run_dir / "failure-test-attempt-1.json").write_text(json.dumps({
                                "error_type": type(first_error).__name__, "message": str(first_error)
                            }, indent=2) + "\n")
                            try:
                                metrics = _logged_call(
                                    seed_run_dir / "test-evaluation.log",
                                    evaluate_checkpoint,
                                    candidate["checkpoint"],
                                    data_root=args.data_root,
                                    task=task,
                                    model_name=model_name,
                                    method=method,
                                    batch_size=32,
                                    nsamples=2,
                                    num_workers=args.workers,
                                    device_name="tpu",
                                    datasets=datasets,
                                    predictions_path=prediction_path,
                                )
                            except Exception as second_error:
                                (seed_run_dir / "failure-test-attempt-2.json").write_text(json.dumps({
                                    "error_type": type(second_error).__name__, "message": str(second_error)
                                }, indent=2) + "\n")
                                raise
                        metrics["checkpoint_sha256"] = checkpoint_hash
                        eval_path.write_text(json.dumps(metrics, indent=2) + "\n")
                        os.chmod(eval_path, 0o600)
                    test_results.append(metrics)
                    all_selected.append(
                        {
                            "task": task,
                            "model": model_name,
                            "method": method,
                            "seed": candidate["seed"],
                            "best_epoch": candidate["best_epoch"],
                            "validation_auroc": candidate["best_val_auroc"],
                            "test_auroc": metrics["auroc"],
                            "test_metric_check": metrics["roc_curve_auc_check"],
                            "test_predictions": metrics["predictions_path"],
                            "checkpoint": candidate["checkpoint"],
                        }
                    )
                    gc.collect()
                test_aucs = np.asarray([item["auroc"] for item in test_results], dtype=float)
                row = {
                    "task": task,
                    "model": model_name,
                    "method": method,
                    "test_auroc_mean_percent": float(test_aucs.mean() * 100),
                    "test_auroc_std_percent": float(test_aucs.std(ddof=1) * 100),
                    "selected_seeds": ",".join(str(item["seed"]) for item in selected),
                    "all_validation_auroc_by_seed": ",".join(
                        f"{item['seed']}:{item['best_val_auroc']:.8f}" for item in validation_by_seed
                    ),
                    "selected_validation_aurocs": ",".join(
                        f"{item['best_val_auroc']:.8f}" for item in selected
                    ),
                    "individual_test_aurocs_percent": ",".join(f"{x * 100:.8f}" for x in test_aucs),
                }
                summary_rows.append(row)
                (root / "selected_checkpoints.json").write_text(
                    json.dumps(all_selected, indent=2) + "\n"
                )
                with (root / "table1_chexpert_auroc.csv").open("w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=list(summary_rows[0]))
                    writer.writeheader()
                    writer.writerows(summary_rows)
                print(
                    f"RESULT task={task}; model={model_name}; method={method}; "
                    f"test_auroc={row['test_auroc_mean_percent']:.2f}±{row['test_auroc_std_percent']:.2f}; "
                    f"selected={row['selected_seeds']}",
                    flush=True,
                )
                del datasets
                gc.collect()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed-start", type=int, default=42)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-task", choices=TASKS, default="Pleural Effusion")
    parser.add_argument("--smoke-model", choices=MODEL_TYPES, default="resnet18")
    parser.add_argument("--smoke-method", choices=METHODS, default="race")
    parser.add_argument("--confirm-original-backbone", action="store_true")
    parser.add_argument("--confirm-model-preflights", action="store_true")
    parser.add_argument("--runtime-root")
    args = parser.parse_args()
    _configure_external_runtime(args.runtime_root or args.output_dir)
    if args.smoke:
        run_smoke(args)
    else:
        run_full(args)


if __name__ == "__main__":
    main()
