from __future__ import annotations

from io import BytesIO
from pathlib import Path
import json
import os
import re
import shutil

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from skimage.io import imread, imsave
from skimage.metrics import structural_similarity
from skimage.transform import resize


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
HF_DATA = ROOT / "hf" / "data"
OUT = ROOT / "hf_preprocessed"
OUT_IMAGES = OUT / "preproc_224x224"
CSV_SOURCE = Path(os.environ.get("CXR_CSV_SOURCE_DIR", REPO / "datafiles" / "chexpert"))
CSV_OUT = REPO / "datafiles" / "chexpert"
STAGED_IMAGES = Path(os.environ["CXR_STAGED_IMAGES"]) if os.environ.get("CXR_STAGED_IMAGES") else None
SPLITS = ("train", "val", "test")


def normalize_to_hf(path: str) -> str:
    name = re.sub(r"^preproc_224x224/", "", path)
    name = name.replace(".jpg.png", ".jpg")
    m = re.fullmatch(r"(patient\d+)_(study\d+)_(view\d+_.*\.jpg)", name)
    if not m:
        raise ValueError(f"Unexpected preprocessed path shape: {name}")
    return f"CheXpert-v1.0-small/train/{m.group(1)}/{m.group(2)}/{m.group(3)}"


def preprocess(raw: bytes) -> np.ndarray:
    image = imread(BytesIO(raw))
    if image.ndim != 2:
        raise ValueError(f"Expected grayscale source image, got shape {image.shape}")
    return resize(image, output_shape=(224, 224), preserve_range=True).astype(np.uint8)


def main() -> None:
    OUT_IMAGES.mkdir(parents=True, exist_ok=True)
    CSV_OUT.mkdir(parents=True, exist_ok=True)
    files = sorted(HF_DATA.glob("train-*.parquet"))
    if len(files) != 23:
        raise RuntimeError(f"Expected 23 training Parquet shards, found {len(files)}")

    samples: dict[str, str] = {}
    for split in SPLITS:
        src = CSV_SOURCE / f"chexpert.sample.{split}.csv"
        dst = CSV_OUT / src.name
        if src.resolve() != dst.resolve():
            shutil.copy2(src, dst)
        df = pd.read_csv(dst)
        for rel in df["path_preproc"].astype(str):
            samples[normalize_to_hf(rel)] = rel

    path_to_index: dict[str, int] = {}
    total_rows = 0
    for file in files:
        pf = pq.ParquetFile(file)
        t = pf.read(columns=["Path"])
        for path in t.column("Path").to_pylist():
            if path in path_to_index:
                raise RuntimeError(f"Duplicate HF path: {path}")
            path_to_index[path] = total_rows
            total_rows += 1
    if len(samples) != 71_096:
        raise RuntimeError(f"Expected 71,096 selected paths, got {len(samples)}")
    missing = {path_to_index[path]: rel for path, rel in samples.items()}
    OUT_IMAGES.mkdir(parents=True, exist_ok=True)

    existing_verify: dict[int, str] = {}
    if STAGED_IMAGES is not None:
        for path, rel in samples.items():
            source = STAGED_IMAGES / Path(rel).name
            if source.exists():
                existing_verify[path_to_index[path]] = rel
                if len(existing_verify) == 32:
                    break
    wanted = set(missing) | set(existing_verify)
    global_offset = 0
    extracted = 0
    verified = 0
    exact_matches = 0
    comparison_ssim: list[float] = []
    selected_metadata = []
    for file in files:
        pf = pq.ParquetFile(file)
        for rg_index in range(pf.num_row_groups):
            n = pf.metadata.row_group(rg_index).num_rows
            start, stop = global_offset, global_offset + n
            group_rows = [r for r in wanted if start <= r < stop]
            if group_rows:
                t = pf.read_row_group(rg_index, columns=["Path", "image"])
                paths = t.column("Path").to_pylist()
                image_struct = t.column("image").combine_chunks()
                raw_bytes = image_struct.field("bytes").to_pylist()
                for row in group_rows:
                    local = row - start
                    hf_path = paths[local]
                    rel = missing.get(row, existing_verify.get(row))
                    image = preprocess(raw_bytes[local])
                    if row in missing:
                        destination = OUT / rel
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        imsave(destination, image)
                        extracted += 1
                    if row in existing_verify:
                        old = STAGED_IMAGES / Path(rel).name
                        expected = imread(old)
                        if expected.shape != image.shape:
                            raise RuntimeError(f"HF/staged image shape mismatch for {hf_path}")
                        exact_matches += int(np.array_equal(expected, image))
                        comparison_ssim.append(float(structural_similarity(expected, image, data_range=255)))
                        verified += 1
            global_offset = stop

    if extracted != len(missing):
        raise RuntimeError(f"Extracted {extracted} of {len(missing)} missing selected images")
    missing_on_disk = [rel for rel in samples.values() if not (OUT / rel).exists()]
    if missing_on_disk:
        raise RuntimeError(f"{len(missing_on_disk)} selected output paths are missing")

    manifest = {
        "hf_train_rows": total_rows,
        "selected_paths": len(samples),
        "linked_existing": 0,
        "extracted_missing": extracted,
        "existing_images_compared": verified,
        "existing_images_exact_pixel_matches": exact_matches,
        "existing_images_mean_ssim": float(np.mean(comparison_ssim)) if comparison_ssim else None,
        "existing_images_min_ssim": float(np.min(comparison_ssim)) if comparison_ssim else None,
        "selected_images_complete": True,
    }
    (ROOT / "preparation.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest))


if __name__ == "__main__":
    main()
