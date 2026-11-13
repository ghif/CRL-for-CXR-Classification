"""Stage only the CheXpert images needed for the paper's Table 1 tasks.

Reads official labels and ZIP central directories with read-only GCS byte ranges,
then writes resized 224x224 grayscale PNGs and minimal manifests to a local
output directory. Requires an authenticated ``gcloud`` CLI.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import struct
import subprocess
import threading
import time
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import requests
from PIL import Image
from skimage.transform import resize


TASKS = ("Pleural Effusion", "Cardiomegaly")
SPLITS = ("train", "validate", "test")
ARCHIVES = (
    "CheXpert-v1.0 batch 2 (train 1).zip",
    "CheXpert-v1.0 batch 3 (train 2).zip",
    "CheXpert-v1.0 batch 4 (train 3).zip",
)
ZIP_PREFIX = "CheXpert-v1.0/train/"
RACE_VALUES = {"White": 0, "Asian": 1, "Black": 2}
_TOKEN_LOCK = threading.Lock()
_TOKEN = ""
_TOKEN_TIME = 0.0
_THREAD_LOCAL = threading.local()


@dataclass(frozen=True)
class ZipEntry:
    object_uri: str
    relative_path: str
    zip_name: str
    offset: int
    compressed_size: int
    uncompressed_size: int
    crc32: int
    method: int
    filename_length: int
    central_extra_length: int


def _access_token(force_refresh: bool = False) -> str:
    global _TOKEN, _TOKEN_TIME
    with _TOKEN_LOCK:
        if force_refresh or not _TOKEN or time.monotonic() - _TOKEN_TIME > 2700:
            try:
                _TOKEN = subprocess.check_output(
                    ["gcloud", "auth", "print-access-token"], text=True
                ).strip()
            except subprocess.CalledProcessError:
                raise RuntimeError("Could not obtain a GCS access token from gcloud") from None
            _TOKEN_TIME = time.monotonic()
        return _TOKEN


def _gcloud_range(uri: str, start: int, end: int) -> bytes:
    if not uri.startswith("gs://"):
        raise ValueError("Expected a GCS URI")
    bucket, object_name = uri[5:].split("/", 1)
    from urllib.parse import quote

    url = "https://storage.googleapis.com/" + quote(bucket + "/" + object_name, safe="/")
    if not hasattr(_THREAD_LOCAL, "session"):
        _THREAD_LOCAL.session = requests.Session()
    session = _THREAD_LOCAL.session
    for attempt in range(4):
        response = session.get(
            url,
            headers={
                "Authorization": "Bearer " + _access_token(force_refresh=(attempt > 0)),
                "Range": f"bytes={start}-{end}",
                "Accept-Encoding": "identity",
            },
            stream=True,
            timeout=(30, 180),
        )
        if response.status_code == 401 and attempt < 3:
            response.close()
            continue
        if response.status_code in (429, 500, 502, 503, 504) and attempt < 3:
            response.close()
            time.sleep(2**attempt)
            continue
        if response.status_code != 206:
            response.close()
            raise RuntimeError(f"GCS byte-range read failed (HTTP {response.status_code})")
        content_range = response.headers.get("Content-Range", "")
        if not content_range.startswith(f"bytes {start}-"):
            response.close()
            raise RuntimeError("GCS returned an unexpected byte range")
        payload = b"".join(response.iter_content(chunk_size=1024 * 1024))
        response.close()
        if len(payload) > end - start + 1:
            raise RuntimeError("GCS returned more bytes than requested")
        return payload
    raise RuntimeError("GCS byte-range read exhausted retries")


def _zip_directory(uri: str) -> list[ZipEntry]:
    tail = subprocess.run(
        ["gcloud", "storage", "cat", "--range", "-65536", uri],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if tail.returncode:
        raise RuntimeError(f"GCS ZIP directory read failed (exit {tail.returncode})")
    tail_bytes = tail.stdout
    eocd_pos = tail_bytes.rfind(b"PK\x05\x06")
    if eocd_pos < 0:
        raise ValueError("ZIP end-of-central-directory record not found")
    _, _, _, _, _, directory_size, directory_offset, _ = struct.unpack_from(
        "<4s4H2LH", tail_bytes, eocd_pos
    )
    if directory_size == 0xFFFFFFFF or directory_offset == 0xFFFFFFFF:
        locator_pos = tail_bytes.rfind(b"PK\x06\x07", 0, eocd_pos)
        if locator_pos < 0:
            raise ValueError("ZIP64 locator not found")
        _, _, zip64_offset, _ = struct.unpack_from("<4sLQL", tail_bytes, locator_pos)
        zip64_end = _gcloud_range(uri, zip64_offset, zip64_offset + 55)
        values = struct.unpack_from("<4sQ2H2I4Q", zip64_end)
        directory_size, directory_offset = values[-2:]

    directory = _gcloud_range(
        uri, directory_offset, directory_offset + directory_size - 1
    )
    entries: list[ZipEntry] = []
    pos = 0
    while pos < len(directory):
        fields = struct.unpack_from("<4s6H3I5H2I", directory, pos)
        if fields[0] != b"PK\x01\x02":
            raise ValueError("Invalid ZIP central-directory entry")
        method = fields[4]
        crc32 = fields[7]
        compressed_size, uncompressed_size = fields[8], fields[9]
        filename_len, extra_len, comment_len = fields[10:13]
        filename_start = pos + 46
        name = directory[filename_start : filename_start + filename_len].decode(
            "utf-8", "replace"
        )
        extra_start = filename_start + filename_len
        extra = directory[extra_start : extra_start + extra_len]
        local_offset = fields[16]

        if (
            compressed_size == 0xFFFFFFFF
            or uncompressed_size == 0xFFFFFFFF
            or local_offset == 0xFFFFFFFF
        ):
            cursor = 0
            while cursor + 4 <= len(extra):
                tag, size = struct.unpack_from("<HH", extra, cursor)
                cursor += 4
                value = extra[cursor : cursor + size]
                cursor += size
                if tag != 1:
                    continue
                value_pos = 0
                if uncompressed_size == 0xFFFFFFFF:
                    uncompressed_size = struct.unpack_from("<Q", value, value_pos)[0]
                    value_pos += 8
                if compressed_size == 0xFFFFFFFF:
                    compressed_size = struct.unpack_from("<Q", value, value_pos)[0]
                    value_pos += 8
                if local_offset == 0xFFFFFFFF:
                    local_offset = struct.unpack_from("<Q", value, value_pos)[0]
                break

        if name.lower().endswith((".jpg", ".jpeg")):
            _, separator, relative_name = name.partition("/")
            if separator and relative_name:
                entries.append(
                    ZipEntry(
                        object_uri=uri,
                        relative_path=relative_name,
                        zip_name=name,
                        offset=local_offset,
                        compressed_size=compressed_size,
                        uncompressed_size=uncompressed_size,
                        crc32=crc32,
                        method=method,
                        filename_length=filename_len,
                        central_extra_length=extra_len,
                    )
                )
        pos += 46 + filename_len + extra_len + comment_len
    return entries


def _read_embedded_train_csv(prefix: str) -> tuple[pd.DataFrame, str]:
    uri = prefix + "CheXpert-v1.0 batch 1 (validate & csv).zip"
    # The label file is not a JPEG and therefore is not returned by the image
    # directory reader; parse just the small central-directory metadata here.
    tail = subprocess.run(
        ["gcloud", "storage", "cat", "--range", "-65536", uri],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if tail.returncode:
        raise RuntimeError(f"GCS label-directory read failed (exit {tail.returncode})")
    t = tail.stdout
    end_pos = t.rfind(b"PK\x05\x06")
    _, _, _, _, _, size, offset, _ = struct.unpack_from("<4s4H2LH", t, end_pos)
    if size == 0xFFFFFFFF or offset == 0xFFFFFFFF:
        locator = t.rfind(b"PK\x06\x07", 0, end_pos)
        _, _, zip64_offset, _ = struct.unpack_from("<4sLQL", t, locator)
        zip64_end = _gcloud_range(uri, zip64_offset, zip64_offset + 55)
        values = struct.unpack_from("<4sQ2H2I4Q", zip64_end)
        size, offset = values[-2:]
    directory = _gcloud_range(uri, offset, offset + size - 1)
    pos = 0
    label_entry = None
    while pos < len(directory):
        fields = struct.unpack_from("<4s6H3I5H2I", directory, pos)
        filename_len, extra_len, comment_len = fields[10:13]
        name_start = pos + 46
        name = directory[name_start : name_start + filename_len].decode("utf-8", "replace")
        if name.endswith("/train.csv") or name == "train.csv":
            label_entry = (fields[4], fields[8], fields[9], fields[16], fields[7])
            break
        pos += 46 + filename_len + extra_len + comment_len
    if label_entry is None:
        raise ValueError("Original train.csv not found in source ZIP")
    method, compressed_size, uncompressed_size, local_offset, expected_crc32 = label_entry
    header = _gcloud_range(uri, local_offset, local_offset + 29)
    local_fields = struct.unpack("<4s5H3I2H", header)
    data_offset = local_offset + 30 + local_fields[-2] + local_fields[-1]
    compressed = _gcloud_range(uri, data_offset, data_offset + compressed_size - 1)
    csv_bytes = zlib.decompress(compressed, -15) if method == 8 else compressed
    if len(csv_bytes) != uncompressed_size or zlib.crc32(csv_bytes) & 0xFFFFFFFF != expected_crc32:
        raise ValueError("Embedded original train.csv failed ZIP size/CRC verification")
    return pd.read_csv(io.BytesIO(csv_bytes)), hashlib.sha256(csv_bytes).hexdigest()


def _read_demo(prefix: str) -> tuple[pd.DataFrame, str, int]:
    result = subprocess.run(
        ["gcloud", "storage", "cat", prefix + "CHEXPERT DEMO.xlsx"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(f"GCS demographic metadata read failed (exit {result.returncode})")
    contents = result.stdout
    demo = pd.read_excel(io.BytesIO(contents))
    renamed = demo.rename(
        columns={"PATIENT": "patient_id", "GENDER": "sex", "PRIMARY_RACE": "race", "ETHNICITY": "ethnicity"}
    )
    return renamed, hashlib.sha256(contents).hexdigest(), len(contents)


def _source_object_metadata(uri: str) -> dict[str, str | int]:
    result = subprocess.run(
        ["gcloud", "storage", "objects", "describe", uri, "--format=json"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        text=True,
    )
    if result.returncode:
        raise RuntimeError(f"GCS source-object metadata read failed (exit {result.returncode})")
    item = json.loads(result.stdout)
    required = ("name", "size", "generation", "crc32c_hash", "md5_hash")
    if any(not item.get(key) for key in required):
        raise ValueError("GCS source-object metadata is incomplete")
    return {
        "name": str(item["name"]),
        "size_bytes": int(item["size"]),
        "generation": str(item["generation"]),
        "crc32c": str(item["crc32c_hash"]),
        "md5": str(item["md5_hash"]),
    }


def _data_quality(rows: pd.DataFrame, smoke: bool) -> dict:
    result = {
        "rows_after_project_cohort_filters": int(len(rows)),
        "duplicate_image_paths": int(rows["Path"].duplicated().sum()),
        "label_values_by_split": {},
        "task_counts_by_split": {},
        "group_cells_by_task_split": {},
        "split_leakage_by_task": {},
        "cooccurrence_by_split": {},
        "attribute_mapping": {
            "race_label": {"0": "White", "1": "Asian", "2": "Black"},
            "sex_label": {"0": "Male", "1": "Female"},
            "view_label": {"0": "AP", "1": "PA"},
        },
    }
    for split_name in SPLITS:
        split_rows = rows[rows["split"] == split_name]
        result["label_values_by_split"][split_name] = {}
        for label in ("No Finding", *TASKS):
            result["label_values_by_split"][split_name][label] = {
                "positive_1": int((split_rows[label] == 1).sum()),
                "negative_0": int((split_rows[label] == 0).sum()),
                "uncertain_minus1": int((split_rows[label] == -1).sum()),
                "missing": int(split_rows[label].isna().sum()),
            }
        result["cooccurrence_by_split"][split_name] = {
            "no_finding_and_pleural_effusion_positive": int(
                ((split_rows["No Finding"] == 1) & (split_rows["Pleural Effusion"] == 1)).sum()
            ),
            "no_finding_and_cardiomegaly_positive": int(
                ((split_rows["No Finding"] == 1) & (split_rows["Cardiomegaly"] == 1)).sum()
            ),
            "both_diseases_positive": int(
                ((split_rows["Pleural Effusion"] == 1) & (split_rows["Cardiomegaly"] == 1)).sum()
            ),
        }

    group_values = {
        "race_label": (0, 1, 2),
        "sex": ("Male", "Female"),
        "AP/PA": ("AP", "PA"),
    }
    expected_table3 = {
        "Pleural Effusion": {
            "train": (6514, 31015), "validate": (1086, 5049), "test": (3316, 15510)
        },
        "Cardiomegaly": {
            "train": (6514, 9353), "validate": (1086, 1548), "test": (3316, 4889)
        },
    }
    for task in TASKS:
        task_mask = ((rows["No Finding"] == 1) | (rows[task] == 1)) & ~(
            (rows["No Finding"] == 1) & (rows[task] == 1)
        )
        task_rows = rows[task_mask]
        result["task_counts_by_split"][task] = {}
        result["group_cells_by_task_split"][task] = {}
        for split_name in SPLITS:
            subset = task_rows[task_rows["split"] == split_name]
            negative_count = int((subset["No Finding"] == 1).sum())
            positive_count = int((subset[task] == 1).sum())
            result["task_counts_by_split"][task][split_name] = {
                "no_finding": negative_count,
                "disease_positive": positive_count,
                "n": int(len(subset)),
            }
            if not smoke and (negative_count, positive_count) != expected_table3[task][split_name]:
                raise ValueError("Filtered task counts do not match paper Table 3")
            result["group_cells_by_task_split"][task][split_name] = {}
            for column, values in group_values.items():
                cells = {}
                for value in values:
                    group = subset[subset[column] == value]
                    cells[str(value)] = {
                        "no_finding": int((group["No Finding"] == 1).sum()),
                        "disease_positive": int((group[task] == 1).sum()),
                    }
                result["group_cells_by_task_split"][task][split_name][column] = cells

        result["split_leakage_by_task"][task] = {}
        task_rows = task_rows.copy()
        task_rows["study_identity"] = task_rows["patient_id"] + "/" + task_rows["Path"].str.split("/").str[3]
        for identity_column in ("patient_id", "study_identity"):
            split_count = task_rows.groupby(identity_column)["split"].nunique()
            result["split_leakage_by_task"][task][identity_column] = {
                "identities_in_multiple_splits": int((split_count > 1).sum()),
                "records_on_leaking_identities": int(
                    task_rows[identity_column].isin(split_count[split_count > 1].index).sum()
                ),
            }
    return result


def _selected_rows(
    train: pd.DataFrame, demo: pd.DataFrame, split_csv: Path, smoke: bool, seed: int
) -> pd.DataFrame:
    train.index.name = "index"
    split = pd.read_csv(split_csv).set_index("index")
    rows = pd.concat([train, split], axis=1)
    rows = rows[rows["split"].notna()].copy()
    rows["patient_id"] = rows["Path"].str.split("/").str[2]
    rows = rows.merge(
        demo[["patient_id", "sex", "race", "ethnicity"]], on="patient_id", how="inner"
    )
    rows.loc[rows["race"].str.contains("Black", na=False), "race"] = "Black"
    rows.loc[rows["race"].str.contains("White", na=False), "race"] = "White"
    rows.loc[rows["race"].str.contains("Asian", na=False), "race"] = "Asian"
    rows = rows[rows["race"].isin(RACE_VALUES)].copy()
    rows = rows[rows["ethnicity"].isin(["Non-Hispanic/Non-Latino", "Not Hispanic"])]
    rows = rows[rows["Frontal/Lateral"] == "Frontal"].copy()
    rows["race_label"] = rows["race"].map(RACE_VALUES).astype("int64")
    if not rows["sex"].isin(["Male", "Female"]).all():
        raise ValueError("Selected records include unsupported sex labels")
    if not rows["AP/PA"].isin(["AP", "PA"]).all():
        raise ValueError("Selected records include unsupported view labels")

    task_masks: dict[str, pd.Series] = {}
    for task in TASKS:
        task_masks[task] = (
            (rows["No Finding"] == 1) | (rows[task] == 1)
        ) & ~((rows["No Finding"] == 1) & (rows[task] == 1))
    if smoke:
        task = "Pleural Effusion"
        task_rows = rows[task_masks[task]].copy()
        chosen = []
        quotas = {"train": 96, "validate": 32, "test": 32}
        for split_name, quota in quotas.items():
            split_rows = task_rows[task_rows["split"] == split_name]
            for is_positive in (False, True):
                label_col = task if is_positive else "No Finding"
                candidates = split_rows[split_rows[label_col] == 1]
                chosen.append(candidates.sample(n=min(quota, len(candidates)), random_state=seed))
        return pd.concat(chosen).drop_duplicates("Path")

    union_mask = task_masks[TASKS[0]] | task_masks[TASKS[1]]
    return rows[union_mask].copy()


def _local_name(path: str) -> str:
    parts = path.split("/")
    if len(parts) != 5 or not path.startswith(ZIP_PREFIX):
        raise ValueError("Unexpected official CheXpert image path")
    return "_".join(parts[2:]) + ".png"


def _image_bytes(entry: ZipEntry) -> bytes:
    uri = entry.object_uri
    # Include a conservative allowance for local-header extras in the same range
    # request. Most entries have no extras; an unusually large extra is retried
    # with an exact two-range read.
    range_end = entry.offset + 30 + entry.filename_length + 4096 + entry.compressed_size - 1
    block = _gcloud_range(uri, entry.offset, range_end)
    if len(block) < 30:
        raise ValueError("Incomplete ZIP local header")
    local = struct.unpack_from("<4s5H3I2H", block)
    if local[0] != b"PK\x03\x04" or local[3] != entry.method or local[2] & 1:
        raise ValueError("Unsupported or invalid ZIP image header")
    name_len, extra_len = local[-2:]
    if 30 + name_len <= len(block):
        local_name_bytes = block[30 : 30 + name_len]
    else:
        local_name_bytes = _gcloud_range(uri, entry.offset + 30, entry.offset + 29 + name_len)
    if local_name_bytes.decode("utf-8", "replace") != entry.zip_name:
        raise ValueError("ZIP local filename does not match its central-directory entry")
    payload_start = 30 + name_len + extra_len
    payload_end = payload_start + entry.compressed_size
    if payload_end > len(block):
        header = _gcloud_range(uri, entry.offset, entry.offset + 29)
        local = struct.unpack("<4s5H3I2H", header)
        name_len, extra_len = local[-2:]
        payload_start = 30 + name_len + extra_len
        payload_end = payload_start + entry.compressed_size
        block = _gcloud_range(
            uri, entry.offset + payload_start, entry.offset + payload_end - 1
        )
        payload_start = 0
        payload_end = len(block)
    compressed = block[payload_start:payload_end]
    if len(compressed) != entry.compressed_size:
        raise ValueError("Incomplete ZIP image payload")
    if entry.method == 0:
        image = compressed
    elif entry.method == 8:
        image = zlib.decompress(compressed, -15)
    else:
        raise ValueError(f"Unsupported ZIP compression method {entry.method}")
    if len(image) != entry.uncompressed_size:
        raise ValueError("ZIP image size does not match the central directory")
    if zlib.crc32(image) & 0xFFFFFFFF != entry.crc32:
        raise ValueError("ZIP image CRC does not match the central directory")
    return image


def _valid_preprocessed(path: Path) -> bool:
    try:
        with Image.open(path) as image:
            if image.size != (224, 224) or image.mode != "L":
                return False
            image.verify()
        with Image.open(path) as image:
            pixels = np.asarray(image)
        return (
            pixels.shape == (224, 224)
            and pixels.dtype == np.uint8
            and int(pixels.min()) >= 0
            and int(pixels.max()) <= 255
        )
    except Exception:
        return False


def _stage_one(image_path: str, entry: ZipEntry, output_root: Path) -> None:
    target = output_root / "preproc_224x224" / _local_name(image_path)
    try:
        # Always read and CRC-check the selected source entry, including when
        # resuming an existing stage.
        source = _image_bytes(entry)
        if target.is_file() and _valid_preprocessed(target):
            return
        target.unlink(missing_ok=True)
        with Image.open(io.BytesIO(source)) as original:
            original.load()
            gray = np.asarray(original.convert("L"))
        if gray.ndim != 2 or gray.size == 0:
            raise ValueError("Expected a nonempty grayscale source image")
        resized = resize(gray, output_shape=(224, 224), preserve_range=True)
        if not np.isfinite(resized).all() or resized.min() < 0 or resized.max() > 255:
            raise ValueError("Resized pixels are outside the expected intensity range")
        target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        tmp = target.with_name(target.stem + f".{os.getpid()}.tmp.png")
        Image.fromarray(resized.astype(np.uint8)).save(tmp, format="PNG")
        os.chmod(tmp, 0o600)
        os.replace(tmp, target)
        if not _valid_preprocessed(target):
            raise ValueError("Preprocessed image failed shape/mode/intensity verification")
    except Exception as exc:
        # Avoid logging patient/image paths or raw content in error messages.
        raise RuntimeError(type(exc).__name__) from None


def _write_manifests(rows: pd.DataFrame, output_root: Path) -> dict[str, str]:
    metadata_dir = output_root / "datafiles" / "chexpert"
    metadata_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    file_hashes = {}
    for split_name, short_name in (("train", "train"), ("validate", "val"), ("test", "test")):
        split_rows = rows[rows["split"] == split_name].copy()
        split_rows["path_preproc"] = split_rows["Path"].map(
            lambda p: "preproc_224x224/" + _local_name(p)
        )
        keep = [
            "path_preproc",
            "No Finding",
            "Cardiomegaly",
            "Pleural Effusion",
            "sex",
            "AP/PA",
            "race_label",
        ]
        path = metadata_dir / f"chexpert.sample.{short_name}.csv"
        split_rows[keep].to_csv(path, index=False)
        os.chmod(path, 0o600)
        file_hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return file_hashes


def _verify_staged_images(rows: pd.DataFrame, output_root: Path) -> int:
    verified = 0
    for image_path in rows["Path"]:
        staged_path = output_root / "preproc_224x224" / _local_name(image_path)
        if not _valid_preprocessed(staged_path):
            raise RuntimeError("A required preprocessed image is missing or invalid")
        verified += 1
    return verified


def prepare(args: argparse.Namespace) -> None:
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(output_root, 0o700)
    prefix = args.gcs_prefix.rstrip("/") + "/"
    source_uris = [
        prefix + "CheXpert-v1.0 batch 1 (validate & csv).zip",
        *(prefix + archive for archive in ARCHIVES),
        prefix + "CHEXPERT DEMO.xlsx",
    ]
    source_metadata = {
        uri.rsplit("/", 1)[-1]: _source_object_metadata(uri) for uri in source_uris
    }
    train, train_sha256 = _read_embedded_train_csv(prefix)
    demo, demo_sha256, demo_size = _read_demo(prefix)
    split_path = Path(args.split_csv)
    split_membership = pd.read_csv(split_path)
    if split_membership["index"].duplicated().any():
        raise ValueError("Split mapping contains duplicate row indices")
    rows = _selected_rows(train, demo, split_path, args.smoke, args.seed)
    if rows.empty:
        raise ValueError("No selected CheXpert rows")

    data_quality = _data_quality(rows, args.smoke)
    if data_quality["duplicate_image_paths"]:
        raise ValueError("Selected manifests contain duplicate image paths")
    if not args.smoke:
        for task in TASKS:
            for column, values in {
                "race_label": (0, 1, 2),
                "sex": ("Male", "Female"),
                "AP/PA": ("AP", "PA"),
            }.items():
                for value in values:
                    cell = data_quality["group_cells_by_task_split"][task]["train"][column][str(value)]
                    if cell["no_finding"] == 0 or cell["disease_positive"] == 0:
                        raise ValueError("An invariant training group lacks one of the task classes")

    manifests = _write_manifests(rows, output_root)
    split_sha256 = hashlib.sha256(split_path.read_bytes()).hexdigest()
    selected_paths = set(rows["Path"])
    required_keys = {path[len(ZIP_PREFIX) :] for path in selected_paths}
    entries: dict[str, ZipEntry] = {}
    for archive in ARCHIVES:
        for entry in _zip_directory(prefix + archive):
            if entry.relative_path in required_keys:
                entries[entry.relative_path] = entry
    missing = required_keys - entries.keys()
    if missing:
        raise RuntimeError(f"{len(missing)} selected images missing from source archives")

    preflight = {
        "scope": "smoke" if args.smoke else "full_table1",
        "source_object_metadata": source_metadata,
        "source_train_csv_rows": int(len(train)),
        "source_train_csv_sha256": train_sha256,
        "demo_workbook_size_bytes": demo_size,
        "demo_workbook_sha256": demo_sha256,
        "split_csv_sha256": split_sha256,
        "split_mapping_rows": int(len(split_membership)),
        "split_mapping_unique_indices": int(split_membership["index"].nunique()),
        "selected_images": int(len(required_keys)),
        "mapped_images": int(len(entries)),
        "missing_images": int(len(missing)),
        "manifest_sha256": manifests,
        "data_quality": data_quality,
    }
    preflight_path = output_root / "preflight.json"
    preflight_path.write_text(json.dumps(preflight, indent=2) + "\n")
    os.chmod(preflight_path, 0o600)

    if args.verify_only:
        print(json.dumps({"selected_images": len(required_keys), "mapped_images": len(entries), "verified_only": True}))
        return

    failures = 0
    completed = 0
    max_pending = max(args.workers * 4, args.workers)
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        pending = set()
        for image_path in rows["Path"]:
            relative = image_path[len(ZIP_PREFIX) :]
            pending.add(executor.submit(_stage_one, image_path, entries[relative], output_root))
            if len(pending) >= max_pending:
                done = next(as_completed(pending))
                pending.remove(done)
                try:
                    done.result()
                except Exception:
                    failures += 1
                completed += 1
                if completed % 500 == 0:
                    print(f"staged {completed}/{len(rows)} images; failures={failures}", flush=True)
        for future in as_completed(pending):
            try:
                future.result()
            except Exception:
                failures += 1
            completed += 1
            if completed % 500 == 0 or completed == len(rows):
                print(f"staged {completed}/{len(rows)} images; failures={failures}", flush=True)
    if failures:
        raise RuntimeError(f"Failed to stage {failures} of {len(rows)} selected images")

    verified_pngs = _verify_staged_images(rows, output_root)
    summary = {
        **preflight,
        "row_counts_by_split": {str(k): int(v) for k, v in rows["split"].value_counts().items()},
        "image_size": [224, 224],
        "image_mode": "L",
        "zip_entry_crc32_verified_images": len(required_keys),
        "verified_png_count": verified_pngs,
        "complete": True,
    }
    summary_path = output_root / "staging_manifest.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    os.chmod(summary_path, 0o600)
    print(json.dumps({"selected_images": len(required_keys), "complete": True}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gcs-prefix", default="gs://external-cxr-dataset/chexpert")
    parser.add_argument(
        "--split-csv",
        default="datafiles/chexpert/chexpert_split_2021_08_20.csv",
    )
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    prepare(args)


if __name__ == "__main__":
    main()
