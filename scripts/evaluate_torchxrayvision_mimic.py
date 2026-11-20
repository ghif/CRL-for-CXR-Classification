#!/usr/bin/env python3
"""Evaluate TorchXRayVision DenseNet-121 (CheXpert, PadChest, MIMIC-CXR) on MIMIC-CXR (6 pathologies) on TorchTPU."""

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
import torch
from skimage.io import imread
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset
import torchxrayvision as xrv


REPO = Path(__file__).resolve().parents[1]
DEFAULT_MIMIC_ROOT = Path("/mnt/data/dataset/mimic_cxr")
DEFAULT_OUTPUT_ROOT = REPO / "chexpert/runs/xrv_densenet121_mimic_cxr"
DEFAULT_WEIGHTS_CACHE = REPO / "chexpert/runs/xrv_densenet121_chexpert_padchest/weights_cache"

WEIGHTS = {
    "chexpert": "densenet121-res224-chex",
    "padchest": "densenet121-res224-pc",
    "mimic_ch": "densenet121-res224-mimic_ch",
}

TASKS = [
    ("Atelectasis", "Atelectasis"),
    ("Cardiomegaly", "Cardiomegaly"),
    ("Pleural Effusion", "Effusion"),
    ("Consolidation", "Consolidation"),
    ("Pneumothorax", "Pneumothorax"),
    ("Edema", "Edema"),
]

PATTERNS = {
    "Atelectasis": r"\batelecta(?:sis|tic)\b",
    "Cardiomegaly": r"\b(?:cardiomegaly|enlarged cardiac silhouette|enlargement of the cardiac silhouette|heart size is (?:mildly |moderately |severely )?enlarged|enlarged heart)\b",
    "Pleural Effusion": r"\b(?:pleural effusions?|effusions?)\b",
    "Consolidation": r"\bconsolidat(?:ion|ive)\b",
    "Pneumothorax": r"\bpneumothora(?:x|ces)\b",
    "Edema": r"\b(?:pulmonary edema|interstitial edema|alveolar edema|vascular congestion|pulmonary congestion)\b",
}
NEG_RE = re.compile(
    r"\b(?:no|without|negative for|free of|absent|absence of|resolved|resolution of|no evidence of|no convincing evidence of|not seen|rules? out|excluded?)\b",
    re.I,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def extract_labels(report_text: str) -> dict[str, int]:
    sentences = re.split(r"[.;\n]+", str(report_text).lower())
    out = {k: 0 for k in PATTERNS}
    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue
        for k, pat in PATTERNS.items():
            m = re.search(pat, sent)
            if m:
                prefix = sent[: m.start()]
                suffix = sent[m.end() : m.end() + 25]
                if NEG_RE.search(prefix) or re.search(r"\b(?:is|are) not seen\b|\bnot excluded\b|\bexcluded\b", suffix):
                    continue
                out[k] = 1
    return out


def build_mimic_validate_df(mimic_root: Path) -> pd.DataFrame:
    csv_path = mimic_root / "mimic_cxr_aug_validate.csv"
    img_root = mimic_root / "official_data_iccv_final"
    df = pd.read_csv(csv_path)
    rows = []
    for _, r in df.iterrows():
        imgs = ast.literal_eval(r["image"])
        studies = list(dict.fromkeys("/".join(p.split("/")[:4]) for p in imgs))
        txts = ast.literal_eval(r["text"])
        study_to_txt = dict(zip(studies, txts))
        ap_set = set(ast.literal_eval(r["AP"]))
        pa_set = set(ast.literal_eval(r["PA"]))
        for p in sorted(ap_set | pa_set):
            fp = img_root / p
            if not fp.exists():
                continue
            st = "/".join(p.split("/")[:4])
            lbls = extract_labels(study_to_txt[st])
            rows.append(
                {
                    "subject_id": r["subject_id"],
                    "relative_path": p,
                    "view": "AP" if p in ap_set else "PA",
                    "image_path": str(fp),
                    **lbls,
                }
            )
    out_df = pd.DataFrame(rows)
    manifest_out = mimic_root / "validate_frontal_6pathologies.csv"
    out_df.drop(columns=["image_path"]).to_csv(manifest_out, index=False)
    return out_df


class MimicXRVDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        self.paths = df["image_path"].tolist()
        self.targets = df[[t for t, _ in TASKS]].to_numpy(dtype=np.int64)
        self.resizer = xrv.datasets.XRayResizer(224, engine="skimage")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        img = imread(self.paths[index])
        if img.ndim == 3:
            img = img[..., 0]
        img = img.astype(np.float32, copy=False)[None, :, :]
        img = xrv.datasets.normalize(img, maxval=255.0)
        img = self.resizer(img)
        return img, self.targets[index]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mimic-root", type=Path, default=DEFAULT_MIMIC_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--weights-cache", type=Path, default=DEFAULT_WEIGHTS_CACHE)
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

    df = build_mimic_validate_df(args.mimic_root)
    dataset = MimicXRVDataset(df)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        worker_init_fn=lambda _: torch.set_num_threads(1),
        persistent_workers=args.num_workers > 0,
    )

    args.output_root.mkdir(parents=True, exist_ok=True)
    args.weights_cache.mkdir(parents=True, exist_ok=True)

    summary: dict = {
        "device": str(device),
        "batch_size": args.batch_size,
        "compile_model": args.compile_model,
        "torchxrayvision_version": importlib.metadata.version("torchxrayvision"),
        "examples": int(len(df)),
        "tasks": [t for t, _ in TASKS],
        "evaluations": {},
    }
    csv_rows = []

    for src_name, weight_name in WEIGHTS.items():
        model = xrv.models.DenseNet(weights=weight_name, cache_dir=str(args.weights_cache))
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

        all_targets: list[np.ndarray] = []
        all_scores: list[np.ndarray] = []
        t0 = time.perf_counter()
        with torch.inference_mode():
            for x, targets in loader:
                actual_bs = len(targets)
                if device.type == "tpu" and actual_bs < args.batch_size:
                    pad = x[:1].expand(args.batch_size - actual_bs, *x.shape[1:])
                    x = torch.cat((x, pad), dim=0)
                x = x.to(device)
                if device.type == "tpu":
                    with torch.autocast("tpu", dtype=torch.bfloat16):
                        out = model(x)
                else:
                    out = model(x)
                all_scores.append(out[:actual_bs, columns].float().cpu().numpy())
                all_targets.append(targets.numpy())
        elapsed = time.perf_counter() - t0

        y = np.concatenate(all_targets)
        scores = np.concatenate(all_scores)
        metrics: dict = {}
        aurocs = []
        auprcs = []
        pred_df = pd.DataFrame({"relative_path": df["relative_path"], "view": df["view"]})

        for idx, (task_name, head) in enumerate(TASKS):
            yt = y[:, idx]
            ys = scores[:, idx]
            pos = int((yt == 1).sum())
            neg = int((yt == 0).sum())
            auroc = float(roc_auc_score(yt, ys))
            auprc = float(average_precision_score(yt, ys))
            aurocs.append(auroc)
            auprcs.append(auprc)
            metrics[task_name] = {
                "xrv_head": head,
                "examples": int(len(yt)),
                "positive": pos,
                "negative": neg,
                "prevalence": pos / len(yt),
                "auroc": auroc,
                "auprc": auprc,
            }
            slug = task_name.lower().replace(" ", "_")
            pred_df[f"target_{slug}"] = yt
            pred_df[f"score_{slug}"] = ys
            csv_rows.append(
                {
                    "pair": f"{src_name}_to_mimic_cxr",
                    "source": src_name,
                    "target": "mimic_cxr",
                    "regime": "in_domain" if src_name == "mimic_ch" else "cross_domain",
                    "task": task_name,
                    "examples": int(len(yt)),
                    "positive": pos,
                    "negative": neg,
                    "prevalence": round(pos / len(yt), 6),
                    "auroc": round(auroc, 6),
                    "auprc": round(auprc, 6),
                }
            )

        metrics["_macro"] = {
            "macro_auroc": float(np.mean(aurocs)),
            "macro_auprc": float(np.mean(auprcs)),
        }
        pred_df.to_csv(args.output_root / f"predictions_{src_name}_to_mimic_cxr.csv", index=False)

        summary["evaluations"][f"{src_name}_to_mimic_cxr"] = {
            "source": src_name,
            "target": "mimic_cxr",
            "regime": "in_domain" if src_name == "mimic_ch" else "cross_domain",
            "weights": weight_name,
            "weights_sha256": weights_sha,
            "elapsed_seconds": round(elapsed, 3),
            "images_per_second": round(len(df) / elapsed, 2),
            "metrics": metrics,
        }
        print(
            json.dumps(
                {
                    "pair": f"{src_name}_to_mimic_cxr",
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
