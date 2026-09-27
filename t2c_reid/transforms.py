"""Image transforms for SigLIP 2-backed person ReID.

The transforms preserve the complete tall person crop at a fixed, patch-aligned
2:1 resolution and reuse the checkpoint processor's normalization statistics.
The image encoder dispatches the resulting BCHW tensor to the loaded
Transformers SigLIP format.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

from PIL import Image
import torch
from torchvision import transforms
from torchvision.transforms import InterpolationMode

# (height, width): 28 x 14 patch14 tokens, or 392 patches total.
DEFAULT_IMAGE_SIZE = (392, 196)


@dataclass(frozen=True)
class ImageTransformConfig:
    image_size: tuple[int, int]
    mean: tuple[float, float, float]
    std: tuple[float, float, float]
    training: bool = False
    flip_prob: float = 0.0
    color_jitter: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    crop_padding: int = 0
    erase_prob: float = 0.0
    erase_scale: tuple[float, float] = (0.02, 0.2)
    erase_ratio: tuple[float, float] = (0.3, 3.3)
    grayscale_prob: float = 0.0

    def __post_init__(self) -> None:
        _validated_image_size(self.image_size)
        if len(self.mean) != 3 or len(self.std) != 3:
            raise ValueError("image normalization mean and std must contain three values")
        if any(not math.isfinite(value) for value in (*self.mean, *self.std)):
            raise ValueError("image normalization mean and std must be finite")
        if any(value <= 0.0 for value in self.std):
            raise ValueError("image normalization std values must be positive")
        for value, name in (
            (self.flip_prob, "flip_prob"),
            (self.erase_prob, "erase_prob"),
            (self.grayscale_prob, "grayscale_prob"),
        ):
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if len(self.color_jitter) != 4:
            raise ValueError("color_jitter must contain four values")
        if any(not math.isfinite(value) or value < 0.0 for value in self.color_jitter[:3]):
            raise ValueError("color_jitter brightness, contrast, and saturation must be finite and non-negative")
        if not math.isfinite(self.color_jitter[3]) or not 0.0 <= self.color_jitter[3] <= 0.5:
            raise ValueError("color_jitter hue must be in [0, 0.5]")
        if len(self.erase_scale) != 2 or any(not math.isfinite(value) for value in self.erase_scale):
            raise ValueError("erase_scale must contain two finite values")
        if not 0.0 < self.erase_scale[0] <= self.erase_scale[1] <= 1.0:
            raise ValueError("erase_scale must satisfy 0 < min <= max <= 1")
        if len(self.erase_ratio) != 2 or any(not math.isfinite(value) for value in self.erase_ratio):
            raise ValueError("erase_ratio must contain two finite values")
        if not 0.0 < self.erase_ratio[0] <= self.erase_ratio[1]:
            raise ValueError("erase_ratio must satisfy 0 < min <= max")
        if self.crop_padding < 0:
            raise ValueError("crop_padding must be non-negative")


TRAIN_FLIP_PROB = 0.5
TRAIN_COLOR_JITTER = (0.2, 0.2, 0.2, 0.05)
TRAIN_CROP_PADDING = 10
TRAIN_ERASE_PROB = 0.5
TRAIN_ERASE_SCALE = (0.02, 0.2)
TRAIN_ERASE_RATIO = (0.3, 3.3)
TRAIN_GRAYSCALE_PROB = 0.0


def _processor_normalization(image_processor: Any) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    mean = getattr(image_processor, "image_mean", None)
    std = getattr(image_processor, "image_std", None)
    if mean is None or std is None:
        raise ValueError(
            "image_processor must expose image_mean and image_std so ReID "
            "transforms can normalize like the SigLIP 2 pretraining pipeline"
        )
    normalized_mean = tuple(float(value) for value in mean)
    normalized_std = tuple(float(value) for value in std)
    if len(normalized_mean) != 3 or len(normalized_std) != 3:
        raise ValueError("image_processor image_mean and image_std must contain three values")
    return normalized_mean, normalized_std


def _validated_image_size(image_size: tuple[int, int]) -> tuple[int, int]:
    if len(image_size) != 2:
        raise ValueError("image_size must be a (height, width) pair")
    result = (int(image_size[0]), int(image_size[1]))
    if result[0] < 1 or result[1] < 1:
        raise ValueError("image dimensions must be positive")
    return result


class Siglip2ImageTransform:
    def __init__(
        self,
        image_processor: Any,
        image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
    ):
        self.image_processor = image_processor
        self.image_size = _validated_image_size(image_size)
        mean, std = _processor_normalization(image_processor)
        self.native_config = ImageTransformConfig(self.image_size, mean, std)
        self._pipeline = transforms.Compose(
            [
                transforms.Resize(self.image_size, interpolation=InterpolationMode.BILINEAR),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )

    def __call__(self, image: Image.Image) -> torch.Tensor:
        return self._pipeline(image)


class Siglip2TrainImageTransform:
    def __init__(
        self,
        image_processor: Any,
        *,
        image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
        flip_prob: float = TRAIN_FLIP_PROB,
        color_jitter: tuple[float, float, float, float] = TRAIN_COLOR_JITTER,
        crop_padding: int = TRAIN_CROP_PADDING,
        erase_prob: float = TRAIN_ERASE_PROB,
        erase_scale: tuple[float, float] = TRAIN_ERASE_SCALE,
        erase_ratio: tuple[float, float] = TRAIN_ERASE_RATIO,
        grayscale_prob: float = TRAIN_GRAYSCALE_PROB,
    ):
        if crop_padding < 0:
            raise ValueError("crop_padding must be non-negative")
        self.image_processor = image_processor
        self.image_size = _validated_image_size(image_size)
        mean, std = _processor_normalization(image_processor)
        self.native_config = ImageTransformConfig(
            image_size=self.image_size,
            mean=mean,
            std=std,
            training=True,
            flip_prob=flip_prob,
            color_jitter=color_jitter,
            crop_padding=crop_padding,
            erase_prob=erase_prob,
            erase_scale=erase_scale,
            erase_ratio=erase_ratio,
            grayscale_prob=grayscale_prob,
        )
        steps: list[Any] = [
            transforms.RandomHorizontalFlip(p=flip_prob),
            transforms.ColorJitter(
                brightness=color_jitter[0],
                contrast=color_jitter[1],
                saturation=color_jitter[2],
                hue=color_jitter[3],
            ),
        ]
        if grayscale_prob > 0.0:
            steps.append(transforms.RandomGrayscale(p=grayscale_prob))
        steps.append(transforms.Resize(self.image_size, interpolation=InterpolationMode.BILINEAR))
        if crop_padding > 0:
            steps.append(transforms.Pad(crop_padding))
            steps.append(transforms.RandomCrop(self.image_size))
        steps.append(transforms.ToTensor())
        steps.append(transforms.Normalize(mean, std))
        self._pipeline = transforms.Compose(steps)
        self.random_erasing = transforms.RandomErasing(
            p=erase_prob,
            scale=erase_scale,
            ratio=erase_ratio,
            value=0.0,
        )

    def __call__(self, image: Image.Image) -> torch.Tensor:
        return self.random_erasing(self._pipeline(image))
