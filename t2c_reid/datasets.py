"""PyTorch dataset helpers for Image-to-Image ReID samples."""

from __future__ import annotations

import random
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import torch
from PIL import Image

from t2c_reid.data import ReIDSample
from t2c_reid.native import native_extension as _native
from t2c_reid.transforms import ImageTransformConfig

ImageTransform = Callable[[Image.Image], torch.Tensor]
MIN_IDENTITIES_PER_BATCH = 2
DEFAULT_INSTANCES_PER_IDENTITY = 2
RUST_DATA_BACKEND = "rust"
PYTHON_DATA_BACKEND = "python"
SUPPORTED_DATA_BACKENDS = (RUST_DATA_BACKEND, PYTHON_DATA_BACKEND)


@dataclass(frozen=True)
class ReIDImageDatasetConfig:
    samples: Sequence[ReIDSample]
    person_id_map: Mapping[int, int]
    camera_id_map: Mapping[int, int]
    transform: ImageTransform


@dataclass(frozen=True)
class ReIDImageItem:
    image: torch.Tensor
    person_id: int
    camera_id: int
    original_person_id: int
    original_camera_id: int


@dataclass(frozen=True)
class ReIDImageRecord:
    image_path: str
    person_id: int
    camera_id: int
    original_person_id: int
    original_camera_id: int


@dataclass(frozen=True)
class ReIDMetadataDatasetConfig:
    samples: Sequence[ReIDSample]
    person_id_map: Mapping[int, int]
    camera_id_map: Mapping[int, int]
    transform: ImageTransformConfig


@dataclass(frozen=True)
class ReIDImageBatch:
    images: torch.Tensor
    person_ids: torch.Tensor
    camera_ids: torch.Tensor
    original_person_ids: tuple[int, ...]
    original_camera_ids: tuple[int, ...]

    def pin_memory(self) -> ReIDImageBatch:
        return ReIDImageBatch(
            images=self.images.pin_memory(),
            person_ids=self.person_ids.pin_memory(),
            camera_ids=self.camera_ids.pin_memory(),
            original_person_ids=self.original_person_ids,
            original_camera_ids=self.original_camera_ids,
        )


class ReIDImageDataset(torch.utils.data.Dataset):
    def __init__(self, config: ReIDImageDatasetConfig):
        self._config = config

    def __len__(self) -> int:
        return len(self._config.samples)

    @property
    def person_ids(self) -> tuple[int, ...]:
        return tuple(self._mapped_person_id(sample) for sample in self._config.samples)

    @property
    def camera_ids(self) -> tuple[int, ...]:
        return tuple(
            _map_value(self._config.camera_id_map, sample.camera_id, "camera_id")
            for sample in self._config.samples
        )

    def __getitem__(self, index: int) -> ReIDImageItem:
        sample = self._config.samples[index]
        return ReIDImageItem(
            image=self._load_image(sample),
            person_id=self._mapped_person_id(sample),
            camera_id=_map_value(
                self._config.camera_id_map, sample.camera_id, "camera_id"
            ),
            original_person_id=sample.person_id,
            original_camera_id=sample.camera_id,
        )

    def _load_image(self, sample: ReIDSample) -> torch.Tensor:
        with Image.open(sample.image_path) as image:
            return self._config.transform(image.convert("RGB"))

    def _mapped_person_id(self, sample: ReIDSample) -> int:
        return _map_value(self._config.person_id_map, sample.person_id, "person_id")


class ReIDMetadataDataset(torch.utils.data.Dataset):
    def __init__(self, config: ReIDMetadataDatasetConfig):
        self._config = config

    def __len__(self) -> int:
        return len(self._config.samples)

    @property
    def person_ids(self) -> tuple[int, ...]:
        return tuple(self._mapped_person_id(sample) for sample in self._config.samples)

    @property
    def camera_ids(self) -> tuple[int, ...]:
        return tuple(
            _map_value(self._config.camera_id_map, sample.camera_id, "camera_id")
            for sample in self._config.samples
        )

    @property
    def transform_config(self) -> ImageTransformConfig:
        return self._config.transform

    def __getitem__(self, index: int) -> ReIDImageRecord:
        sample = self._config.samples[index]
        return ReIDImageRecord(
            image_path=str(sample.image_path),
            person_id=self._mapped_person_id(sample),
            camera_id=_map_value(
                self._config.camera_id_map, sample.camera_id, "camera_id"
            ),
            original_person_id=sample.person_id,
            original_camera_id=sample.camera_id,
        )

    def _mapped_person_id(self, sample: ReIDSample) -> int:
        return _map_value(self._config.person_id_map, sample.person_id, "person_id")


@dataclass(frozen=True)
class RustReIDBatchCollator:
    transform: ImageTransformConfig
    threads: int = 1

    def __post_init__(self) -> None:
        if self.threads < 1:
            raise ValueError("Rust data threads must be positive")

    def __call__(self, items: Sequence[ReIDImageRecord]) -> ReIDImageBatch:
        if not items:
            raise ValueError("cannot collate an empty ReID batch")
        config = self.transform
        batch_seed = (
            int(
                torch.randint(
                    0, torch.iinfo(torch.int64).max, (), dtype=torch.int64
                ).item()
            )
            if config.training
            else 0
        )
        images = torch.from_numpy(
            _native.load_image_batch(
                [item.image_path for item in items],
                batch_seed,
                config.image_size[0],
                config.image_size[1],
                list(config.mean),
                list(config.std),
                config.training,
                config.flip_prob,
                list(config.color_jitter),
                config.crop_padding,
                config.erase_prob,
                list(config.erase_scale),
                list(config.erase_ratio),
                config.grayscale_prob,
                self.threads,
            )
        )
        return ReIDImageBatch(
            images=images,
            person_ids=torch.tensor(
                [item.person_id for item in items], dtype=torch.long
            ),
            camera_ids=torch.tensor(
                [item.camera_id for item in items], dtype=torch.long
            ),
            original_person_ids=tuple(item.original_person_id for item in items),
            original_camera_ids=tuple(item.original_camera_id for item in items),
        )


class IdentityBalancedBatchSampler(torch.utils.data.Sampler[list[int]]):
    """PK sampler: ``batch_size // K`` identities with ``K`` images each.

    With ``camera_labels`` the ``K`` images of an identity are drawn round-robin
    over its cameras in shuffled order, so every identity with images from
    several cameras (for PRCC, several clothing states) contributes
    cross-camera positives to each batch instead of relying on chance.
    """

    def __init__(
        self,
        labels: Sequence[int],
        batch_size: int,
        instances_per_identity: int = DEFAULT_INSTANCES_PER_IDENTITY,
        camera_labels: Sequence[int] | None = None,
    ):
        self._labels = tuple(labels)
        self._batch_size = batch_size
        self._instances_per_identity = instances_per_identity
        self._identities_per_batch = _identities_per_batch(
            batch_size, instances_per_identity
        )
        self._groups = _eligible_identity_groups(self._labels, instances_per_identity)
        _validate_identity_groups(
            self._groups, self._identities_per_batch, instances_per_identity
        )
        if camera_labels is not None and len(camera_labels) != len(self._labels):
            raise ValueError("camera_labels must contain one camera per sample label")
        self._camera_groups = (
            None
            if camera_labels is None
            else _identity_camera_groups(self._groups, tuple(camera_labels))
        )

    def __iter__(self):
        for _ in range(len(self)):
            yield self._sample_batch()

    def __len__(self) -> int:
        return max(1, len(self._labels) // self._batch_size)

    def _sample_batch(self) -> list[int]:
        labels = random.sample(tuple(self._groups), self._identities_per_batch)
        return [index for label in labels for index in self._sample_instances(label)]

    def _sample_instances(self, label: int) -> list[int]:
        if self._camera_groups is None:
            return random.sample(self._groups[label], self._instances_per_identity)
        return _camera_round_robin(
            self._camera_groups[label], self._instances_per_identity
        )


def build_person_id_map(samples: Sequence[ReIDSample]) -> dict[int, int]:
    return _build_index_map(sample.person_id for sample in samples)


def build_camera_id_map(samples: Sequence[ReIDSample]) -> dict[int, int]:
    return _build_index_map(sample.camera_id for sample in samples)


def collate_reid_batches(items: Sequence[ReIDImageItem]) -> ReIDImageBatch:
    return ReIDImageBatch(
        images=torch.stack([item.image for item in items]),
        person_ids=torch.tensor([item.person_id for item in items], dtype=torch.long),
        camera_ids=torch.tensor([item.camera_id for item in items], dtype=torch.long),
        original_person_ids=tuple(item.original_person_id for item in items),
        original_camera_ids=tuple(item.original_camera_id for item in items),
    )


def require_data_backend(backend: str) -> str:
    if backend not in SUPPORTED_DATA_BACKENDS:
        raise ValueError(
            f"unsupported data backend {backend!r}; expected one of {SUPPORTED_DATA_BACKENDS}"
        )
    return backend


def _identities_per_batch(batch_size: int, instances_per_identity: int) -> int:
    if batch_size < MIN_IDENTITIES_PER_BATCH * instances_per_identity:
        raise ValueError(
            "batch_size must allow at least two identities with positive pairs"
        )
    if batch_size % instances_per_identity != 0:
        raise ValueError("batch_size must be divisible by instances_per_identity")
    return batch_size // instances_per_identity


def _eligible_identity_groups(
    labels: Sequence[int], instances_per_identity: int
) -> dict[int, list[int]]:
    groups: dict[int, list[int]] = {}
    for index, label in enumerate(labels):
        groups.setdefault(label, []).append(index)
    insufficient = {
        label: len(indices)
        for label, indices in groups.items()
        if len(indices) < instances_per_identity
    }
    if insufficient:
        raise ValueError(
            f"every training identity requires at least {instances_per_identity} images; "
            f"insufficient identity counts: {insufficient}"
        )
    return groups


def _identity_camera_groups(
    groups: Mapping[int, list[int]], camera_labels: Sequence[int]
) -> dict[int, list[list[int]]]:
    camera_groups: dict[int, list[list[int]]] = {}
    for label, indices in groups.items():
        by_camera: dict[int, list[int]] = {}
        for index in indices:
            by_camera.setdefault(camera_labels[index], []).append(index)
        camera_groups[label] = [by_camera[camera] for camera in sorted(by_camera)]
    return camera_groups


def _camera_round_robin(cameras: Sequence[list[int]], count: int) -> list[int]:
    pools = [random.sample(indices, len(indices)) for indices in cameras]
    random.shuffle(pools)
    selected: list[int] = []
    while len(selected) < count:
        pools = [pool for pool in pools if pool]
        for pool in pools[: count - len(selected)]:
            selected.append(pool.pop())
    return selected


def _validate_identity_groups(
    groups: Mapping[int, list[int]],
    identities_per_batch: int,
    instances_per_identity: int,
) -> None:
    if len(groups) < identities_per_batch:
        raise ValueError(
            "identity-balanced sampling requires at least "
            f"{identities_per_batch} identities with {instances_per_identity} images each"
        )


def _build_index_map(values) -> dict[int, int]:
    return {value: index for index, value in enumerate(sorted(set(values)))}


def _map_value(mapping: Mapping[int, int], value: int, name: str) -> int:
    if value not in mapping:
        raise KeyError(f"{name} {value} is missing from the index map")
    return mapping[value]
