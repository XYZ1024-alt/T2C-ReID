"""Dataset protocol helpers for Market-1501, MSMT17, and PRCC."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
import random
import re

MARKET_PATTERN = re.compile(r"^(?P<pid>-?\d+)_c(?P<cam>\d+)s\d+_\d+_\d+\.jpg$")
MSMT_PATTERN = re.compile(r"^\d{4}_\d{3}_(?P<cam>\d{2})_.+\.jpg$")

MARKET_SPLIT_DIRS = {
    "train": "bounding_box_train",
    "query": "query",
    "gallery": "bounding_box_test",
}

MSMT_MANIFESTS = {
    "train": ("list_train.txt", "train"),
    "val": ("list_val.txt", "train"),
    "query": ("list_query.txt", "test"),
    "gallery": ("list_gallery.txt", "test"),
}

# PRCC (RGB modality): cameras A and B share clothes; C shows changed clothes.
# CAL-style evaluation uses every test/A image as the gallery, test/C as the
# cross-clothes query (primary metric), and test/B as the same-clothes query.
PRCC_TRAIN_PATTERN = re.compile(r"^(?P<cam>[ABC])_cropped_rgb\d+\.jpg$")
PRCC_CAMERA_IDS = {"A": 1, "B": 2, "C": 3}
PRCC_GALLERY_CAMERA = "A"
PRCC_SAME_CLOTHES_CAMERA = "B"
PRCC_CROSS_CLOTHES_CAMERA = "C"
PRCC_TEST_SPLIT_CAMERAS = {
    "gallery": PRCC_GALLERY_CAMERA,
    "query_same": PRCC_SAME_CLOTHES_CAMERA,
    "query_cross": PRCC_CROSS_CLOTHES_CAMERA,
}
# Fixed so every tuning run validates on the same held-out identities.
PRCC_HOLDOUT_SEED = 0

JUNK_PERSON_ID = -1


@dataclass(frozen=True)
class ReIDSample:
    image_path: Path
    person_id: int
    camera_id: int
    dataset: str
    split: str


def parse_market_filename(filename: str) -> tuple[int, int]:
    match = MARKET_PATTERN.match(Path(filename).name)
    if match is None:
        raise ValueError(f"Unsupported Market-1501 filename: {filename}")
    return int(match.group("pid")), int(match.group("cam"))


def parse_msmt17_filename(filename: str) -> int:
    match = MSMT_PATTERN.match(Path(filename).name)
    if match is None:
        raise ValueError(f"Unsupported MSMT17 filename: {filename}")
    return int(match.group("cam"))


def parse_prcc_train_filename(filename: str) -> int:
    match = PRCC_TRAIN_PATTERN.match(Path(filename).name)
    if match is None:
        raise ValueError(f"Unsupported PRCC training filename: {filename}")
    return PRCC_CAMERA_IDS[match.group("cam")]


def load_market_split(root: Path, split: str) -> list[ReIDSample]:
    split_dir = _market_split_dir(root, split)
    samples = [_market_sample(path, split) for path in sorted(split_dir.glob("*.jpg"))]
    return [sample for sample in samples if sample.person_id != JUNK_PERSON_ID]


def load_msmt17_manifest(root: Path, split: str) -> list[ReIDSample]:
    manifest_name, image_dir = _msmt_manifest(split)
    manifest_path = root / manifest_name
    lines = manifest_path.read_text(encoding="utf-8").splitlines()
    return [_msmt_sample(root, image_dir, split, line) for line in lines if line.strip()]


def load_prcc_split(root: Path, split: str) -> list[ReIDSample]:
    """Load one PRCC RGB split from ``root/rgb``.

    ``train`` reads ``rgb/train/<pid>/<cam>_cropped_rgb*.jpg``. The test splits
    ``gallery`` / ``query_same`` / ``query_cross`` read ``rgb/test/{A,B,C}/<pid>``,
    whose filenames carry no camera token; the camera is the directory.
    """
    rgb_root = root / "rgb"
    if split == "train":
        paths = sorted((rgb_root / "train").glob("*/*.jpg"))
        return [_prcc_sample(path, parse_prcc_train_filename(path.name), split) for path in paths]
    if split not in PRCC_TEST_SPLIT_CAMERAS:
        raise ValueError(f"Unsupported PRCC split: {split}")
    camera = PRCC_TEST_SPLIT_CAMERAS[split]
    paths = sorted((rgb_root / "test" / camera).glob("*/*.jpg"))
    return [_prcc_sample(path, PRCC_CAMERA_IDS[camera], split) for path in paths]


@dataclass(frozen=True)
class PRCCHoldoutSplits:
    train: list[ReIDSample]
    gallery: list[ReIDSample]
    query_same: list[ReIDSample]
    query_cross: list[ReIDSample]


def split_prcc_holdout(
    samples: Sequence[ReIDSample],
    identity_count: int,
    *,
    seed: int = PRCC_HOLDOUT_SEED,
) -> PRCCHoldoutSplits:
    """Hold out whole PRCC training identities as an ID-disjoint validation set.

    Held-out identities mirror the test protocol: camera A is the gallery, B the
    same-clothes query, and C the cross-clothes query. The test split is never
    read, so model selection on this holdout does not touch test labels.
    """
    person_ids = sorted({sample.person_id for sample in samples})
    if not 0 < identity_count < len(person_ids):
        raise ValueError(
            f"PRCC holdout identity count must be in [1, {len(person_ids) - 1}], "
            f"got {identity_count}"
        )
    held_out = set(random.Random(seed).sample(person_ids, identity_count))
    train = [sample for sample in samples if sample.person_id not in held_out]
    held_out_samples = [sample for sample in samples if sample.person_id in held_out]
    splits = PRCCHoldoutSplits(
        train=train,
        gallery=_held_out_camera_split(held_out_samples, PRCC_GALLERY_CAMERA, "gallery"),
        query_same=_held_out_camera_split(held_out_samples, PRCC_SAME_CLOTHES_CAMERA, "query_same"),
        query_cross=_held_out_camera_split(
            held_out_samples, PRCC_CROSS_CLOTHES_CAMERA, "query_cross"
        ),
    )
    gallery_ids = {sample.person_id for sample in splits.gallery}
    missing = sorted(pid for pid in held_out if pid not in gallery_ids)
    if missing:
        raise ValueError(f"PRCC holdout identities have no camera-A gallery images: {missing}")
    return splits


def _prcc_sample(path: Path, camera_id: int, split: str) -> ReIDSample:
    return ReIDSample(path, int(path.parent.name), camera_id, "prcc", split)


def _held_out_camera_split(
    samples: Sequence[ReIDSample], camera: str, split: str
) -> list[ReIDSample]:
    camera_id = PRCC_CAMERA_IDS[camera]
    return [replace(sample, split=split) for sample in samples if sample.camera_id == camera_id]


def _market_split_dir(root: Path, split: str) -> Path:
    if split not in MARKET_SPLIT_DIRS:
        raise ValueError(f"Unsupported Market-1501 split: {split}")
    return root / MARKET_SPLIT_DIRS[split]


def _market_sample(path: Path, split: str) -> ReIDSample:
    person_id, camera_id = parse_market_filename(path.name)
    return ReIDSample(path, person_id, camera_id, "market1501", split)


def _msmt_manifest(split: str) -> tuple[str, str]:
    if split not in MSMT_MANIFESTS:
        raise ValueError(f"Unsupported MSMT17 split: {split}")
    return MSMT_MANIFESTS[split]


def _msmt_sample(root: Path, image_dir: str, split: str, line: str) -> ReIDSample:
    relative_path, person_id = _parse_manifest_line(line)
    image_path = root / image_dir / relative_path
    camera_id = parse_msmt17_filename(relative_path.name)
    return ReIDSample(image_path, person_id, camera_id, "msmt17", split)


def _parse_manifest_line(line: str) -> tuple[Path, int]:
    parts = line.split()
    if len(parts) != 2:
        raise ValueError(f"Invalid MSMT17 manifest line: {line}")
    return Path(parts[0]), int(parts[1])
