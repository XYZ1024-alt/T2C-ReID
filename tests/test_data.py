from pathlib import Path
import tempfile
import unittest

from t2c_reid.data import (
    ReIDSample,
    load_market_split,
    load_msmt17_manifest,
    load_prcc_split,
    parse_market_filename,
    parse_msmt17_filename,
    parse_prcc_train_filename,
    split_prcc_holdout,
)


class DataParsingTest(unittest.TestCase):
    def test_parse_market_filename_reads_person_and_camera(self):
        self.assertEqual(parse_market_filename("0002_c3s1_000551_01.jpg"), (2, 3))

    def test_parse_msmt17_filename_reads_camera_token(self):
        self.assertEqual(parse_msmt17_filename("0000_045_12_0303morning_0006_2.jpg"), 12)

    def test_load_market_split_skips_junk_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            split_dir = Path(tmp) / "bounding_box_train"
            split_dir.mkdir()
            (split_dir / "0002_c3s1_000551_01.jpg").touch()
            (split_dir / "-1_c1s1_000401_03.jpg").touch()

            samples = load_market_split(Path(tmp), "train")

        self.assertEqual(
            samples,
            [ReIDSample(split_dir / "0002_c3s1_000551_01.jpg", 2, 3, "market1501", "train")],
        )

    def test_load_msmt17_manifest_uses_manifest_label_and_camera(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "list_query.txt").write_text(
                "0000/0000_000_01_0303morning_0015_0.jpg 7\n",
                encoding="utf-8",
            )

            samples = load_msmt17_manifest(root, "query")

        expected_path = root / "test" / "0000" / "0000_000_01_0303morning_0015_0.jpg"
        self.assertEqual(samples, [ReIDSample(expected_path, 7, 1, "msmt17", "query")])

    def test_parse_prcc_train_filename_maps_camera_letter(self):
        self.assertEqual(parse_prcc_train_filename("C_cropped_rgb553.jpg"), 3)
        with self.assertRaises(ValueError):
            parse_prcc_train_filename("cropped_rgb553.jpg")

    def test_load_prcc_split_reads_camera_from_prefix_or_test_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train_path = root / "rgb" / "train" / "007" / "B_cropped_rgb001.jpg"
            query_path = root / "rgb" / "test" / "C" / "012" / "cropped_rgb002.jpg"
            gallery_path = root / "rgb" / "test" / "A" / "012" / "cropped_rgb003.jpg"
            for path in (train_path, query_path, gallery_path):
                path.parent.mkdir(parents=True)
                path.touch()

            train = load_prcc_split(root, "train")
            query = load_prcc_split(root, "query_cross")
            gallery = load_prcc_split(root, "gallery")
            same_clothes = load_prcc_split(root, "query_same")

        self.assertEqual(train, [ReIDSample(train_path, 7, 2, "prcc", "train")])
        self.assertEqual(query, [ReIDSample(query_path, 12, 3, "prcc", "query_cross")])
        self.assertEqual(gallery, [ReIDSample(gallery_path, 12, 1, "prcc", "gallery")])
        self.assertEqual(same_clothes, [])

    def test_split_prcc_holdout_is_identity_disjoint_and_deterministic(self):
        samples = [
            ReIDSample(Path(f"{pid}/{cam}.jpg"), pid, cam, "prcc", "train")
            for pid in range(10)
            for cam in (1, 2, 3)
        ]

        first = split_prcc_holdout(samples, 3)
        second = split_prcc_holdout(samples, 3)

        held_out = {sample.person_id for sample in first.gallery}
        self.assertEqual(first, second)
        self.assertEqual(len(held_out), 3)
        self.assertFalse(held_out & {sample.person_id for sample in first.train})
        self.assertEqual(len(first.train), 21)
        self.assertEqual({sample.camera_id for sample in first.gallery}, {1})
        self.assertEqual({sample.camera_id for sample in first.query_same}, {2})
        self.assertEqual({sample.camera_id for sample in first.query_cross}, {3})
        self.assertEqual({sample.split for sample in first.query_cross}, {"query_cross"})
        with self.assertRaises(ValueError):
            split_prcc_holdout(samples, 10)

    def test_split_prcc_holdout_requires_cross_clothes_images_but_not_same_clothes(self):
        without_same_clothes = [
            ReIDSample(Path(f"{pid}/{cam}.jpg"), pid, cam, "prcc", "train")
            for pid in range(2)
            for cam in (1, 3)
        ]
        without_cross_clothes = [
            ReIDSample(Path(f"{pid}/{cam}.jpg"), pid, cam, "prcc", "train")
            for pid in range(2)
            for cam in (1, 2)
        ]

        splits = split_prcc_holdout(without_same_clothes, 1)

        self.assertEqual(splits.query_same, [])
        self.assertEqual(len(splits.query_cross), 1)
        with self.assertRaisesRegex(ValueError, "camera-C"):
            split_prcc_holdout(without_cross_clothes, 1)
