#!/usr/bin/env python3
"""Evaluate TorchXRayVision DenseNet-121 (CheXpert & PadChest) cross-domain on TorchTPU."""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.metadata
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from skimage.io import imread
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset
import torchxrayvision as xrv


REPO = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = REPO / "chexpert/runs/xrv_densenet121_chexpert_padchest"
CHEXPERT_CSV = REPO / "datafiles/chexpert/chexpert.sample.test.csv"
CHEXPERT_HF_DIR = Path("/mnt/data/dataset/chexpert/hf/data")
CHEXPERT_IMG_DIR = Path("/mnt/data/dataset/chexpert/hf_preprocessed")
PADCHEST_CSV = Path("/mnt/data/dataset/padchest/metadata/PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv")
PADCHEST_STAGE_DIR = Path("/mnt/data/dataset/padchest/input-stage/images-224")
PADCHEST_MANIFEST = PADCHEST_STAGE_DIR / "padchest-stage-manifest.jsonl"

WEIGHTS = {
    "chexpert": "densenet121-res224-chex",
    "padchest": "densenet121-res224-pc",
}

TASKS = [
    ("Atelectasis", "Atelectasis"),
    ("Cardiomegaly", "Cardiomegaly"),
    ("Pleural Effusion", "Effusion"),
    ("Consolidation", "Consolidation"),
    ("Pneumothorax", "Pneumothorax"),
    ("Edema", "Edema"),
]

PADCHEST_LABEL_MAP = {
    "Atelectasis": {
        "atelectasis",
        "atelectasis basal",
        "laminar atelectasis",
        "lobar atelectasis",
        "round atelectasis",
        "segmental atelectasis",
        "total atelectasis",
    },
    "Cardiomegaly": {"cardiomegaly"},
    "Pleural Effusion": {
        "pleural effusion",
        "loculated pleural effusion",
        "loculated fissural effusion",
    },
    "Consolidation": {"consolidation"},
    "Pneumothorax": {"pneumothorax", "hydropneumothorax"},
    "Edema": {"pulmonary edema"},
}

FRONTAL_PROJECTIONS = {"PA", "AP", "AP_horizontal"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_to_hf_path(path_preproc: str) -> str:
    name = re.sub(r"^preproc_224x224/", "", path_preproc).replace(".jpg.png", ".jpg")
    m = re.fullmatch(r"(patient\d+)_(study\d+)_(view\d+_.*\.jpg)", name)
    if not m:
        raise ValueError(f"Unexpected preprocessed CheXpert path: {path_preproc}")
    return f"CheXpert-v1.0-small/train/{m.group(1)}/{m.group(2)}/{m.group(3)}"


def load_chexpert_test_metadata() -> pd.DataFrame:
    df = pd.read_csv(CHEXPERT_CSV)
    df["hf_path"] = df["path_preproc"].astype(str).map(normalize_to_hf_path)
    wanted_paths = set(df["hf_path"])

    task_cols = [task for task, _ in TASKS]
    hf_records: list[pd.DataFrame] = []
    for shard in sorted(CHEXPERT_HF_DIR.glob("train-*.parquet")):
        table = pq.read_table(shard, columns=["Path", *task_cols]).to_pandas()
        matched = table[table["Path"].isin(wanted_paths)]
        if not matched.empty:
            hf_records.append(matched)

    hf_df = pd.concat(hf_records, ignore_index=True).drop_duplicates(subset=["Path"])
    merged = df.merge(hf_df, left_on="hf_path", right_on="Path", how="inner", suffixes=("_csv", ""))
    if len(merged) != len(df):
        raise RuntimeError(f"Expected {len(df)} matched CheXpert rows in HF parquet, got {len(merged)}")

    for task in task_cols:
        # HF danjacobellis/chexpert encoding: 3=positive(1.0), 2=negative(0.0), 1=uncertain(-1.0), 0=blank(NaN).
        # Binarize using U-Zeroes (positive == 3 -> 1, else 0).
        merged[f"target_{task}"] = (merged[task] == 3).astype(np.int64)

    merged["image_path"] = merged["path_preproc"].astype(str).map(lambda p: str(CHEXPERT_IMG_DIR / p))
    merged["subgroup_view"] = merged["AP/PA"].astype(str)
    merged["subgroup_sex"] = merged["sex"].astype(str)
    return merged


def load_padchest_test_metadata() -> pd.DataFrame:
    manifest_rows = []
    with PADCHEST_MANIFEST.open("r", encoding="utf-8") as stream:
        for line in stream:
            rec = json.loads(line)
            if rec.get("kind") == "item" and rec.get("split") == "test":
                manifest_rows.append((rec["image_id"], rec["relative_path"]))
    manifest_df = pd.DataFrame(manifest_rows, columns=["ImageID", "relative_path"]).drop_duplicates("ImageID")

    pc_df = pd.read_csv(
        PADCHEST_CSV,
        usecols=["ImageID", "Projection", "MethodLabel", "Labels", "PatientSex_DICOM"],
        low_memory=False,
    )
    merged = manifest_df.merge(pc_df, on="ImageID", how="inner")
    merged = merged[merged["Projection"].isin(FRONTAL_PROJECTIONS)].copy().reset_index(drop=True)

    parsed_labels: list[set[str]] = []
    for raw in merged["Labels"].fillna("[]"):
        try:
            items = ast.literal_eval(raw)
            parsed_labels.append({str(x).strip().lower() for x in items})
        except (ValueError, SyntaxError):
            parsed_labels.append(set())

    for task, _ in TASKS:
        synonyms = PADCHEST_LABEL_MAP[task]
        merged[f"target_{task}"] = [1 if (lbls & synonyms) else 0 for lbls in parsed_labels]

    merged["image_path"] = merged["relative_path"].map(lambda p: str(PADCHEST_STAGE_DIR / p))
    merged["subgroup_view"] = merged["Projection"].astype(str)
    merged["subgroup_method"] = merged["MethodLabel"].astype(str)
    return merged


class GrayscaleCXRDataset(Dataset):
    def __init__(self, df: pd.DataFrame, maxval: float):
        self.paths = df["image_path"].tolist()
        self.targets = df[[f"target_{task}" for task, _ in TASKS]].to_numpy(dtype=np.int64)
        self.maxval = maxval

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        img = imread(self.paths[index])
        if img.ndim == 3:
            img = img[..., 0]
        img = img.astype(np.float32, copy=False)[None, :, :]
        img = xrv.datasets.normalize(img, maxval=self.maxval)
        return img, self.targets[index]


def evaluate_pair(
    model: torch.nn.Module,
    columns: list[int],
    df: pd.DataFrame,
    maxval: float,
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    dataset = GrayscaleCXRDataset(df, maxval=maxval)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        worker_init_fn=lambda _: torch.set_num_threads(1),
        persistent_workers=num_workers > 0,
    )

    all_targets: list[np.ndarray] = []
    all_scores: list[np.ndarray] = []
    t0 = time.perf_counter()
    with torch.inference_mode():
        for batch_idx, (x, targets) in enumerate(loader, start=1):
            actual_bs = len(targets)
            if device.type == "tpu" and actual_bs < batch_size:
                pad = x[:1].expand(batch_size - actual_bs, *x.shape[1:])
                x = torch.cat((x, pad), dim=0)
            x = x.to(device)
            if device.type == "tpu":
                with torch.autocast("tpu", dtype=torch.bfloat16):
                    out = model(x)
            else:
                out = model(x)
            scores = out[:actual_bs, columns].float().cpu().numpy()
            all_targets.append(targets.numpy())
            all_scores.append(scores)
            if batch_idx % 100 == 0 or batch_idx == len(loader):
                print(json.dumps({"batch": batch_idx, "total_batches": len(loader)}), flush=True)
    elapsed = time.perf_counter() - t0
    return np.concatenate(all_targets), np.concatenate(all_scores), elapsed


def compute_metrics(y: np.ndarray, scores: np.ndarray) -> dict[str, dict]:
    metrics: dict[str, dict] = {}
    aurocs = []
    auprcs = []
    for idx, (task, xrv_head) in enumerate(TASKS):
        yt = y[:, idx]
        ys = scores[:, idx]
        pos = int((yt == 1).sum())
        neg = int((yt == 0).sum())
        auroc = float(roc_auc_score(yt, ys)) if pos > 0 and neg > 0 else None
        auprc = float(average_precision_score(yt, ys)) if pos > 0 else None
        if auroc is not None:
            aurocs.append(auroc)
        if auprc is not None:
            auprcs.append(auprc)
        metrics[task] = {
            "xrv_head": xrv_head,
            "examples": int(len(yt)),
            "positive": pos,
            "negative": neg,
            "prevalence": pos / len(yt) if len(yt) else 0.0,
            "auroc": auroc,
            "auprc": auprc,
        }
    metrics["_macro"] = {
        "macro_auroc": float(np.mean(aurocs)) if aurocs else None,
        "macro_auprc": float(np.mean(auprcs)) if auprcs else None,
    }
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--device", default="tpu")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--compile-model", action="store_true", default=True)
    parser.add_argument("--no-compile", dest="compile_model", action="store_false")
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == "tpu":
        import torch_tpu  # noqa: F401

        if not torch.accelerator.is_available() or torch.accelerator.current_accelerator().type != "tpu":
            raise RuntimeError("--device tpu requested, but TorchTPU reports no available TPU")
    torch.set_num_threads(32 if device.type == "tpu" else 4)

    args.output_root.mkdir(parents=True, exist_ok=True)
    cache_dir = args.output_root / "weights_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    print(json.dumps({"status": "loading_metadata"}), flush=True)
    datasets_meta = {
        "chexpert": (load_chexpert_test_metadata(), 255.0),
        "padchest": (load_padchest_test_metadata(), 65535.0),
    }
    for name, (df, maxval) in datasets_meta.items():
        print(json.dumps({"dataset": name, "rows": len(df), "maxval": maxval}), flush=True)

    summary: dict = {
        "device": str(device),
        "batch_size": args.batch_size,
        "compile_model": args.compile_model,
        "torchxrayvision_version": importlib.metadata.version("torchxrayvision"),
        "tasks": [t for t, _ in TASKS],
        "evaluations": {},
    }
    csv_rows = []

    for src_name, weight_name in WEIGHTS.items():
        model = xrv.models.DenseNet(weights=weight_name, cache_dir=str(cache_dir))
        model.op_threshs = None
        model.apply_sigmoid = False
        pathologies = list(model.pathologies)
        columns = [pathologies.index(head) for _, head in TASKS]
        weights_path = Path(model.weights_filename_local)
        weights_sha = sha256(weights_path)

        model = model.to(device).eval()
        if device.type == "tpu" and args.compile_model:
            xrv.utils.warning_log["norm_check"] = True
            model = torch.compile(model, backend="tpu", dynamic=False)

        for tgt_name, (df, maxval) in datasets_meta.items():
            pair_key = f"{src_name}_to_{tgt_name}"
            regime = "in_domain" if src_name == tgt_name else "cross_domain"
            print(json.dumps({"status": "evaluating", "pair": pair_key, "regime": regime}), flush=True)
            y, scores, elapsed = evaluate_pair(
                model=model,
                columns=columns,
                df=df,
                maxval=maxval,
                device=device,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
            )
            metrics = compute_metrics(y, scores)
            subgroups = {}
            if tgt_name == "padchest" and "subgroup_method" in df.columns:
                for method_val in sorted(df["subgroup_method"].unique()):
                    mask = (df["subgroup_method"] == method_val).to_numpy()
                    subgroups[f"MethodLabel={method_val}"] = compute_metrics(y[mask], scores[mask])

            pred_df = pd.DataFrame()
            for idx, (task, _) in enumerate(TASKS):
                slug = task.lower().replace(" ", "_")
                pred_df[f"target_{slug}"] = y[:, idx]
                pred_df[f"score_{slug}"] = scores[:, idx]
            pred_path = args.output_root / f"predictions_{pair_key}.csv"
            pred_df.to_csv(pred_path, index=False)

            summary["evaluations"][pair_key] = {
                "source": src_name,
                "target": tgt_name,
                "regime": regime,
                "weights": weight_name,
                "weights_sha256": weights_sha,
                "examples": int(len(df)),
                "elapsed_seconds": round(elapsed, 3),
                "images_per_second": round(len(df) / elapsed, 2) if elapsed > 0 else None,
                "metrics": metrics,
                "subgroups": subgroups,
            }

            for task, _ in TASKS:
                m = metrics[task]
                csv_rows.append(
                    {
                        "pair": pair_key,
                        "source": src_name,
                        "target": tgt_name,
                        "regime": regime,
                        "task": task,
                        "examples": m["examples"],
                        "positive": m["positive"],
                        "negative": m["negative"],
                        "prevalence": round(m["prevalence"], 6),
                        "auroc": round(m["auroc"], 6) if m["auroc"] is not None else None,
                        "auprc": round(m["auprc"], 6) if m["auprc"] is not None else None,
                    }
                )
            print(
                json.dumps(
                    {
                        "pair": pair_key,
                        "macro_auroc": metrics["_macro"]["macro_auroc"],
                        "macro_auprc": metrics["_macro"]["macro_auprc"],
                        "elapsed_s": round(elapsed, 2),
                    }
                ),
                flush=True,
            )

    (args.output_root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    pd.DataFrame(csv_rows).to_csv(args.output_root / "comparison.csv", index=False)
    print(json.dumps({"status": "done", "output_root": str(args.output_root)}), flush=True)


if __name__ == "__main__":
    main()
