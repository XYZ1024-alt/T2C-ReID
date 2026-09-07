import pickle
import tempfile
import unittest
from collections import Counter
from pathlib import Path

import torch
from PIL import Image

from t2c_reid.data import ReIDSample
from t2c_reid.datasets import (
    IdentityBalancedBatchSampler,
    ReIDImageBatch,
    ReIDImageDataset,
    ReIDImageDatasetConfig,
    ReIDImageRecord,
    ReIDMetadataDataset,
    ReIDMetadataDatasetConfig,
    RustReIDBatchCollator,
    build_camera_id_map,
    build_person_id_map,
    collate_reid_batches,
)
from t2c_reid.transforms import ImageTransformConfig


class ReIDImageDatasetTest(unittest.TestCase):
    def test_reid_image_dataset_returns_remapped_and_original_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            image_path = Path(tmp) / "0002_c3s1_000551_01.jpg"
            Image.new("RGB", (2, 2), color="red").save(image_path)
            sample = ReIDSample(image_path, 42, 7, "market1501", "train")
            dataset = ReIDImageDataset(_dataset_config([sample], {42: 0}, {7: 0}))

            batch = dataset[0]

        self.assertTrue(torch.equal(batch.image, torch.ones(3, 2, 2)))
        self.assertEqual(batch.person_id, 0)
        self.assertEqual(batch.camera_id, 0)
        self.assertEqual(batch.original_person_id, 42)
        self.assertEqual(batch.original_camera_id, 7)

    def test_build_index_maps_sorts_unique_values(self):
        samples = [
            ReIDSample(Path("a.jpg"), 9, 3, "market1501", "train"),
            ReIDSample(Path("b.jpg"), 4, 1, "market1501", "train"),
        ]

        self.assertEqual(build_person_id_map(samples), {4: 0, 9: 1})
        self.assertEqual(build_camera_id_map(samples), {1: 0, 3: 1})

    def test_collate_reid_batches_stacks_images_and_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            first = _sample(Path(tmp) / "a.jpg", 9, 3)
            second = _sample(Path(tmp) / "b.jpg", 4, 1)
            dataset = ReIDImageDataset(
                _dataset_config([first, second], {4: 0, 9: 1}, {1: 0, 3: 1})
            )

            batch = collate_reid_batches([dataset[0], dataset[1]])

        self.assertEqual(batch.images.shape, (2, 3, 2, 2))
        self.assertTrue(torch.equal(batch.person_ids, torch.tensor([1, 0])))
        self.assertTrue(torch.equal(batch.camera_ids, torch.tensor([1, 0])))
        self.assertEqual(batch.original_person_ids, (9, 4))
        self.assertEqual(batch.original_camera_ids, (3, 1))

    def test_native_batch_collator_preserves_order_shape_and_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            first = _sample(Path(tmp) / "a.jpg", 9, 3)
            second = _sample(Path(tmp) / "b.jpg", 4, 1)
            dataset = ReIDMetadataDataset(
                ReIDMetadataDatasetConfig(
                    [first, second],
                    {4: 0, 9: 1},
                    {1: 0, 3: 1},
                    _native_transform_config(),
                )
            )
            collator = RustReIDBatchCollator(dataset.transform_config)

            batch = collator([dataset[0], dataset[1]])

        self.assertEqual(batch.images.shape, (2, 3, 4, 2))
        self.assertEqual(batch.images.dtype, torch.float32)
        self.assertTrue(batch.images.is_contiguous())
        self.assertTrue(torch.equal(batch.person_ids, torch.tensor([1, 0])))
        self.assertTrue(torch.equal(batch.camera_ids, torch.tensor([1, 0])))
        self.assertEqual(batch.original_person_ids, (9, 4))
        self.assertEqual(batch.original_camera_ids, (3, 1))

    def test_native_eval_collator_does_not_advance_torch_rng(self):
        with tempfile.TemporaryDirectory() as tmp:
            sample = _sample(Path(tmp) / "a.jpg", 9, 3)
            config = _native_transform_config()
            dataset = ReIDMetadataDataset(
                ReIDMetadataDatasetConfig([sample], {9: 0}, {3: 0}, config)
            )
            torch.manual_seed(123)
            before = torch.get_rng_state().clone()

            RustReIDBatchCollator(config)([dataset[0]])

            self.assertTrue(torch.equal(torch.get_rng_state(), before))

    def test_native_train_collator_replays_fixed_torch_seed(self):
        with tempfile.TemporaryDirectory() as tmp:
            sample = _sample(Path(tmp) / "a.jpg", 9, 3)
            config = _native_transform_config(training=True)
            dataset = ReIDMetadataDataset(
                ReIDMetadataDatasetConfig([sample], {9: 0}, {3: 0}, config)
            )
            collator = RustReIDBatchCollator(config)

            torch.manual_seed(123)
            first = collator([dataset[0]]).images.clone()
            torch.manual_seed(123)
            second = collator([dataset[0]]).images.clone()

        self.assertTrue(torch.equal(first, second))

    def test_native_collator_is_spawn_pickleable_and_reports_corrupt_path(self):
        config = _native_transform_config()
        collator = RustReIDBatchCollator(config)
        restored = pickle.loads(pickle.dumps(collator))
        self.assertEqual(restored, collator)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "broken.jpg"
            path.write_bytes(b"not an image")
            record = ReIDImageRecord(str(path), 0, 0, 1, 1)
            with self.assertRaisesRegex(OSError, "broken.jpg"):
                collator([record])

    def test_native_collator_runs_in_spawn_workers(self):
        with tempfile.TemporaryDirectory() as tmp:
            samples = [
                _sample(Path(tmp) / "a.jpg", 9, 3),
                _sample(Path(tmp) / "b.jpg", 4, 1),
            ]
            config = _native_transform_config()
            dataset = ReIDMetadataDataset(
                ReIDMetadataDatasetConfig(samples, {4: 0, 9: 1}, {1: 0, 3: 1}, config)
            )
            loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=2,
                num_workers=2,
                persistent_workers=True,
                prefetch_factor=2,
                collate_fn=RustReIDBatchCollator(config),
            )

            batch = next(iter(loader))

        self.assertEqual(batch.images.shape, (2, 3, 4, 2))
        self.assertEqual(set(batch.original_person_ids), {4, 9})

    @unittest.skipUnless(
        torch.cuda.is_available(), "pin_memory requires a CUDA runtime"
    )
    def test_reid_batch_pin_memory_pins_tensor_fields(self):
        batch = ReIDImageBatch(
            images=torch.ones(1, 3, 2, 2),
            person_ids=torch.tensor([1]),
            camera_ids=torch.tensor([2]),
            original_person_ids=(10,),
            original_camera_ids=(20,),
        )

        pinned = batch.pin_memory()

        self.assertTrue(pinned.images.is_pinned())
        self.assertTrue(pinned.person_ids.is_pinned())
        self.assertTrue(pinned.camera_ids.is_pinned())
        self.assertEqual(pinned.original_person_ids, (10,))

    def test_native_pipeline_rejects_non_finite_erasing_ranges(self):
        from t2c_reid.native import native_extension

        with self.assertRaisesRegex(ValueError, "erase_scale"):
            ImageTransformConfig(
                image_size=(4, 2),
                mean=(0.5, 0.5, 0.5),
                std=(0.5, 0.5, 0.5),
                erase_scale=(float("nan"), 0.2),
            )
        with self.assertRaisesRegex(ValueError, "erase_ratio"):
            native_extension.load_image_batch(
                ["unused.jpg"],
                0,
                4,
                2,
                [0.5, 0.5, 0.5],
                [0.5, 0.5, 0.5],
                True,
                0.5,
                [0.2, 0.2, 0.2, 0.05],
                1,
                0.5,
                [0.02, 0.2],
                [0.3, float("inf")],
                1,
            )

    def test_native_pipeline_decodes_grayscale_and_rgba_png(self):
        with tempfile.TemporaryDirectory() as tmp:
            gray_path = Path(tmp) / "gray.png"
            rgba_path = Path(tmp) / "rgba.png"
            Image.new("L", (2, 2), color=128).save(gray_path)
            Image.new("RGBA", (2, 2), color=(255, 0, 0, 0)).save(rgba_path)
            config = ImageTransformConfig(
                image_size=(2, 2),
                mean=(0.5, 0.5, 0.5),
                std=(0.5, 0.5, 0.5),
            )
            records = [
                ReIDImageRecord(str(gray_path), 0, 0, 0, 0),
                ReIDImageRecord(str(rgba_path), 1, 1, 1, 1),
            ]

            images = RustReIDBatchCollator(config)(records).images

        self.assertTrue(torch.equal(images[0, 0], images[0, 1]))
        self.assertTrue(torch.equal(images[0, 1], images[0, 2]))
        self.assertTrue(
            torch.allclose(images[1, :, 0, 0], torch.tensor([1.0, -1.0, -1.0]))
        )

    def test_native_eval_resize_stays_within_one_8bit_step(self):
        class Processor:
            image_mean = (0.5, 0.5, 0.5)
            image_std = (0.5, 0.5, 0.5)

        from t2c_reid.transforms import Siglip2ImageTransform

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "pattern.png"
            pixels = (
                torch.arange(37 * 19 * 3, dtype=torch.uint8).reshape(19, 37, 3).numpy()
            )
            Image.fromarray(pixels, mode="RGB").save(path)
            transform = Siglip2ImageTransform(Processor(), image_size=(64, 32))
            python_image = transform(Image.open(path).convert("RGB"))
            dataset = ReIDMetadataDataset(
                ReIDMetadataDatasetConfig(
                    [ReIDSample(path, 1, 1, "market1501", "query")],
                    {1: 0},
                    {1: 0},
                    transform.native_config,
                )
            )

            native_image = RustReIDBatchCollator(transform.native_config)(
                [dataset[0]]
            ).images[0]

        difference = (native_image - python_image).abs()
        self.assertLessEqual(float(difference.max()), 2.0 / 255.0 + 1e-6)
        self.assertLess(float(difference.mean()), 0.0025)

    def test_identity_balanced_batch_sampler_groups_positive_and_negative_pairs(self):
        labels = [0, 0, 0, 1, 1, 1, 2, 2]
        sampler = IdentityBalancedBatchSampler(
            labels, batch_size=4, instances_per_identity=2
        )

        batch = next(iter(sampler))
        counts = Counter(labels[index] for index in batch)

        self.assertEqual(len(counts), 2)
        self.assertEqual(set(counts.values()), {2})

    def test_identity_balanced_batch_sampler_rejects_missing_positive_pairs(self):
        labels = [0, 1, 2, 3]

        with self.assertRaises(ValueError):
            IdentityBalancedBatchSampler(labels, batch_size=4, instances_per_identity=2)

    def test_identity_balanced_batch_sampler_does_not_silently_drop_an_identity(self):
        with self.assertRaisesRegex(ValueError, "insufficient identity counts:.*2: 1"):
            IdentityBalancedBatchSampler(
                [0, 0, 1, 1, 2], batch_size=4, instances_per_identity=2
            )


def _native_transform_config(*, training: bool = False) -> ImageTransformConfig:
    return ImageTransformConfig(
        image_size=(4, 2),
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
        training=training,
        flip_prob=0.5,
        color_jitter=(0.2, 0.2, 0.2, 0.05),
        crop_padding=1,
        erase_prob=0.5,
        erase_scale=(0.02, 0.2),
        erase_ratio=(0.3, 3.3),
    )


def _tensor_transform(image: Image.Image) -> torch.Tensor:
    return torch.ones(3, image.height, image.width)


def _sample(path: Path, person_id: int, camera_id: int) -> ReIDSample:
    Image.new("RGB", (2, 2), color="blue").save(path)
    return ReIDSample(path, person_id, camera_id, "market1501", "train")


def _dataset_config(samples, person_id_map, camera_id_map) -> ReIDImageDatasetConfig:
    return ReIDImageDatasetConfig(
        samples, person_id_map, camera_id_map, _tensor_transform
    )
