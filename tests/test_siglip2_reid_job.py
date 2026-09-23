import random
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch
from PIL import Image
from transformers import SiglipConfig, SiglipModel

from t2c_reid.configuration import TrainingConfig, compose_training_config
from t2c_reid.datasets import ReIDImageBatch, collate_reid_batches
from t2c_reid.jobs.siglip2_reid import (
    BetaSchedule,
    JobDataConfig,
    Siglip2LoadResult,
    Siglip2ReIDTrainingModel,
    StageLRScheduler,
    TransformBundle,
    _extract_features,
    _validate_loaded_siglip2,
    build_training_job,
    load_dataset_bundle,
    load_transformers_siglip2,
)
from t2c_reid.retrieval import IMAGE_ONLY_RETRIEVAL
from t2c_reid.transforms import Siglip2ImageTransform
from tests._siglip2_fakes import (
    FakeSiglip2,
    FakeSiglip2ImageProcessor,
    FakeSiglip2Tokenizer,
)


class Siglip2ReIDJobTest(unittest.TestCase):
    def test_validator_rejects_naflex_model_type(self):
        model = FakeSiglip2()
        model.config.model_type = "siglip2"
        loaded = Siglip2LoadResult(
            model,
            FakeSiglip2ImageProcessor(),
            FakeSiglip2Tokenizer(),
        )

        with self.assertRaisesRegex(ValueError, "model_type 'siglip'"):
            _validate_loaded_siglip2(loaded, (392, 196))

    def test_public_loader_rejects_non_target_checkpoint(self):
        with self.assertRaisesRegex(ValueError, "only supports"):
            load_transformers_siglip2("google/siglip2-so400m-patch16-naflex")

    def test_load_dataset_bundle_rejects_missing_root(self):
        config = JobDataConfig("market1501", Path("missing"))

        with self.assertRaises(FileNotFoundError):
            load_dataset_bundle(config, FakeSiglip2ImageProcessor())

    def test_msmt17_training_split_merges_train_and_val_manifests(self):
        # Standard MSMT17 protocol (TransReID/ SigLIP 2-ReID) trains on
        # list_train.txt + list_val.txt; dropping val loses ~7% of the data.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "list_train.txt").write_text(
                "0000/0000_000_01_0303morning_0015_0.jpg 0\n"
                "0001/0001_000_02_0303morning_0016_0.jpg 1\n",
                encoding="utf-8",
            )
            (root / "list_val.txt").write_text(
                "0000/0000_001_03_0303morning_0017_0.jpg 0\n",
                encoding="utf-8",
            )
            (root / "list_query.txt").write_text(
                "0002/0002_000_01_0303morning_0018_0.jpg 2\n",
                encoding="utf-8",
            )
            (root / "list_gallery.txt").write_text(
                "0002/0002_000_02_0303morning_0019_0.jpg 2\n",
                encoding="utf-8",
            )

            data = load_dataset_bundle(
                JobDataConfig("msmt17", root), FakeSiglip2ImageProcessor()
            )

        self.assertEqual(len(data.train), 3)
        self.assertEqual(data.num_train_ids, 2)
        self.assertTrue(torch.equal(data.identity_counts, torch.tensor([2, 1])))
        self.assertEqual(tuple(data.identity_camera_counts.shape), (2, 3))
        self.assertTrue(
            torch.equal(data.identity_camera_counts.sum(dim=1), data.identity_counts)
        )

    def test_load_dataset_bundle_uses_train_transform_only_for_train_split(self):
        class ConstantTransform:
            def __init__(self, value: float):
                self.value = value

            def __call__(self, image):
                return torch.full((3, image.height, image.width), self.value)

        with tempfile.TemporaryDirectory() as tmp:
            root = _build_market_fixture(Path(tmp))
            transforms = TransformBundle(
                train=ConstantTransform(2.0),
                eval=ConstantTransform(1.0),
            )

            data = load_dataset_bundle(JobDataConfig("market1501", root), transforms)

            self.assertTrue(
                torch.equal(data.train[0].image, torch.full((3, 2, 2), 2.0))
            )
            self.assertTrue(torch.equal(data.query[0].image, torch.ones(3, 2, 2)))
            self.assertTrue(torch.equal(data.gallery[0].image, torch.ones(3, 2, 2)))

    def test_training_statistics_match_python_and_rust_backends(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = _build_market_fixture(Path(tmp))
            processor = FakeSiglip2ImageProcessor()
            transforms = TransformBundle(
                train=Siglip2ImageTransform(processor),
                eval=Siglip2ImageTransform(processor),
            )

            python = load_dataset_bundle(
                JobDataConfig("market1501", root), transforms, backend="python"
            )
            rust = load_dataset_bundle(
                JobDataConfig("market1501", root), transforms, backend="rust"
            )

        self.assertTrue(torch.equal(python.identity_counts, rust.identity_counts))
        self.assertTrue(
            torch.equal(python.identity_camera_counts, rust.identity_camera_counts)
        )

    def test_build_training_job_returns_real_callbacks_with_fake_siglip2(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = _build_market_fixture(Path(tmp))
            job = build_training_job(
                _training_config(root), siglip2_loader=_load_fake_siglip2
            )
            reporter = TrainBatchReporterRecorder()

            train_metrics = job.train_one_epoch(1, reporter)
            metrics = job.validate(1)

        self.assertEqual(len(reporter.batch_reports), 1)
        self.assertIn("loss", train_metrics)
        self.assertIn("alignment_loss", train_metrics)
        self.assertIn("reid_loss", train_metrics)
        self.assertIn("triplet_loss", train_metrics)
        self.assertIn("triplet_active_fraction", train_metrics)
        self.assertIn("tfc_loss", train_metrics)
        self.assertIn("tfc_local_loss", train_metrics)
        self.assertIn("tfc_global_loss", train_metrics)
        self.assertIn("tfc_cross_modal_loss", train_metrics)
        self.assertIn("tfc_cross_camera_loss", train_metrics)
        self.assertIn("tfc_transfer_reg_loss", train_metrics)
        self.assertIn("tfc_cross_camera_coverage", train_metrics)
        self.assertIn("lr", train_metrics)
        self.assertIsNotNone(job.stage_metadata)
        self.assertEqual(job.stage_metadata.get("dataset"), "market1501")
        self.assertEqual(job.stage_metadata.get("data_backend"), "rust")
        self.assertEqual(job.stage_metadata.get("evaluation_backend"), "rust")
        self.assertFalse(job.stage_metadata.get("pin_memory"))
        self.assertFalse(job.stage_metadata.get("persistent_workers"))
        self.assertEqual(job.checkpoint_metadata["schema_version"], 3)
        self.assertEqual(
            job.checkpoint_metadata["tfc_version"],
            "camera_aware_cross_modal_v1",
        )
        self.assertIn("pid_camera_count_fingerprint", job.checkpoint_metadata)
        self.assertEqual(job.checkpoint_metadata["tfc_weight"], 1.0)
        self.assertEqual(job.checkpoint_metadata["beta"], 0.1)
        self.assertEqual(job.checkpoint_metadata["beta_warmup_epochs"], 0)
        self.assertEqual(job.checkpoint_metadata["stage1_epochs"], 0)
        self.assertEqual(job.checkpoint_metadata["stage2_first_epoch"], 1)
        self.assertGreaterEqual(metrics.map, 0.0)
        self.assertIn(1, metrics.cmc)

    def test_fixed_checkpoint_registration_uses_official_input_contract(self):
        model = SiglipModel(
            SiglipConfig(
                text_config={
                    "vocab_size": 32,
                    "hidden_size": 8,
                    "intermediate_size": 16,
                    "num_hidden_layers": 1,
                    "num_attention_heads": 2,
                    "max_position_embeddings": 12,
                    "pad_token_id": 0,
                    "bos_token_id": 2,
                    "eos_token_id": 1,
                    "projection_size": 8,
                },
                vision_config={
                    "hidden_size": 8,
                    "intermediate_size": 16,
                    "num_hidden_layers": 1,
                    "num_attention_heads": 2,
                    "image_size": 8,
                    "patch_size": 2,
                },
            )
        )
        loaded = Siglip2LoadResult(
            model=model,
            image_processor=SimpleNamespace(size={"height": 8, "width": 8}),
            tokenizer=SimpleNamespace(
                bos_token_id=2,
                eos_token_id=1,
                pad_token_id=0,
                padding_side="right",
                model_input_names=["input_ids"],
            ),
        )

        spec = _validate_loaded_siglip2(loaded, (8, 4))

        self.assertEqual(spec.vision_input_format, "fixed_bchw")
        self.assertEqual(spec.patch_count, 8)
        self.assertEqual(spec.max_num_patches, 16)
        self.assertEqual(spec.text_padding_side, "right")
        self.assertFalse(spec.include_bos_token)
        self.assertFalse(spec.mask_text_padding)

        loaded.tokenizer.padding_side = "left"
        with self.assertRaisesRegex(ValueError, "right padding"):
            _validate_loaded_siglip2(loaded, (8, 4))

        loaded.tokenizer.padding_side = "right"
        loaded.tokenizer.model_input_names = ["input_ids", "attention_mask"]
        with self.assertRaisesRegex(ValueError, "input_ids-only"):
            _validate_loaded_siglip2(loaded, (8, 4))

    def test_loaded_processor_must_match_model_patch_budget(self):
        def mismatched_loader(model_name: str) -> Siglip2LoadResult:
            return Siglip2LoadResult(
                FakeSiglip2(hidden_size=8, feature_dim=4),
                FakeSiglip2ImageProcessor(max_num_patches=256),
                FakeSiglip2Tokenizer(),
            )

        with tempfile.TemporaryDirectory() as tmp:
            root = _build_market_fixture(Path(tmp))
            with self.assertRaisesRegex(ValueError, "patch budget mismatch"):
                build_training_job(
                    _training_config(root), siglip2_loader=mismatched_loader
                )

    def test_loaded_tokenizer_special_ids_must_match_model(self):
        class WrongTokenizer(FakeSiglip2Tokenizer):
            bos_token_id = 3

        def mismatched_loader(model_name: str) -> Siglip2LoadResult:
            return Siglip2LoadResult(
                FakeSiglip2(hidden_size=8, feature_dim=4),
                FakeSiglip2ImageProcessor(),
                WrongTokenizer(),
            )

        with tempfile.TemporaryDirectory() as tmp:
            root = _build_market_fixture(Path(tmp))
            with self.assertRaisesRegex(ValueError, "bos_token_id mismatch"):
                build_training_job(
                    _training_config(root), siglip2_loader=mismatched_loader
                )

    def test_non_patch_aligned_image_size_fails_at_startup(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = _build_market_fixture(Path(tmp))
            args = _training_config(root)
            args.image_height = 391
            with self.assertRaisesRegex(ValueError, "divisible"):
                build_training_job(args, siglip2_loader=_load_fake_siglip2)

    def test_gradient_checkpointing_flag_reaches_loaded_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = _build_market_fixture(Path(tmp))
            enabled = build_training_job(
                _training_config(root), siglip2_loader=_load_fake_siglip2
            )
            args = _training_config(root)
            args.gradient_checkpointing = False
            disabled = build_training_job(args, siglip2_loader=_load_fake_siglip2)

        self.assertTrue(
            enabled.model.retrieval_model.image_encoder.siglip2_model.gradient_checkpointing
        )
        self.assertFalse(
            disabled.model.retrieval_model.image_encoder.siglip2_model.gradient_checkpointing
        )

    def test_build_training_job_returns_two_stage_job_when_stage1_epochs_positive(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = _build_market_fixture(Path(tmp))
            args = _training_config(root)
            args.stage1_epochs = 1
            job = build_training_job(args, siglip2_loader=_load_fake_siglip2)

        from scripts.train import TwoStageTrainingJob

        self.assertIsInstance(job, TwoStageTrainingJob)

    def test_build_training_job_rejects_training_split_without_positive_pairs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = _build_market_fixture_without_positive_pairs(Path(tmp))

            with self.assertRaises(ValueError):
                build_training_job(
                    _training_config(root), siglip2_loader=_load_fake_siglip2
                )

    def test_extract_features_passes_configured_retrieval_mode(self):
        model = RetrievalModeRecorder()
        batch = ReIDImageBatch(
            images=torch.ones(2, 3, 2, 2),
            person_ids=torch.tensor([0, 1]),
            camera_ids=torch.tensor([1, 2]),
            original_person_ids=(10, 20),
            original_camera_ids=(1, 2),
        )

        features = _extract_features(
            model,
            [batch],
            torch.device("cpu"),
            retrieval_mode=IMAGE_ONLY_RETRIEVAL,
        )

        self.assertEqual(model.retrieval_modes, [IMAGE_ONLY_RETRIEVAL])
        self.assertEqual(features.person_ids, (10, 20))
        self.assertEqual(features.camera_ids, (1, 2))
        self.assertEqual(tuple(features.features.shape), (2, 4))

    def test_flip_tta_is_off_by_default_and_runs_one_forward_per_batch(self):
        model = RetrievalModeRecorder()
        batch = _asymmetric_batch()

        _extract_features(model, [batch], torch.device("cpu"), IMAGE_ONLY_RETRIEVAL)

        self.assertEqual(len(model.retrieval_modes), 1)

    def test_flip_tta_averages_the_image_and_its_mirror(self):
        model = RetrievalModeRecorder()
        batch = _asymmetric_batch()

        _extract_features(
            model, [batch], torch.device("cpu"), IMAGE_ONLY_RETRIEVAL, flip_tta=True
        )

        self.assertEqual(len(model.seen_images), 2)
        original, flipped = model.seen_images
        self.assertTrue(torch.equal(flipped, torch.flip(original, dims=(3,))))

    def test_flip_tta_feature_is_the_renormalized_mean_of_both_views(self):
        model = DirectionalRetrievalStub()
        batch = _asymmetric_batch()

        plain = _extract_features(
            model, [batch], torch.device("cpu"), IMAGE_ONLY_RETRIEVAL
        )
        averaged = _extract_features(
            model, [batch], torch.device("cpu"), IMAGE_ONLY_RETRIEVAL, flip_tta=True
        )

        # A left-right asymmetric image and its mirror give opposite directions,
        # so the flip average must differ from either view and stay normalized.
        self.assertFalse(torch.allclose(plain.features, averaged.features))
        self.assertTrue(
            torch.allclose(
                averaged.features.norm(dim=1),
                torch.ones(averaged.features.shape[0]),
                atol=1e-6,
            )
        )

    def test_prcc_bundle_uses_cross_clothes_query_and_same_clothes_extra(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = _build_prcc_fixture(Path(tmp))

            data = load_dataset_bundle(
                JobDataConfig("prcc", root), FakeSiglip2ImageProcessor()
            )

        self.assertEqual(data.num_train_ids, 3)
        self.assertEqual(data.num_cameras, 3)
        self.assertEqual(len(data.query), 1)
        self.assertEqual(len(data.gallery), 2)
        self.assertEqual([name for name, _ in data.extra_queries], ["same_clothes"])

    def test_prcc_holdout_validates_on_training_identities_without_test_split(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = _build_prcc_fixture(Path(tmp), with_test=False)

            data = load_dataset_bundle(
                JobDataConfig("prcc", root, validation_holdout_ids=1),
                FakeSiglip2ImageProcessor(),
            )

        self.assertEqual(data.num_train_ids, 2)
        self.assertEqual(len(data.query), 2)
        self.assertEqual(len(data.gallery), 2)

    def test_prcc_validation_reports_same_clothes_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = _build_prcc_fixture(Path(tmp))
            args = _training_config(root)
            args.dataset = "prcc"

            job = build_training_job(args, siglip2_loader=_load_fake_siglip2)
            metrics = job.validate(1)

        self.assertIn(1, metrics.cmc)
        self.assertIn("same_clothes_mAP", metrics.extras)
        self.assertIn("same_clothes_rank_1", metrics.extras)

    def test_fused_validation_also_scores_image_only_retrieval(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = _build_market_fixture(Path(tmp))
            fused_job = build_training_job(
                _training_config(root), siglip2_loader=_load_fake_siglip2
            )
            image_only_args = _training_config(root)
            image_only_args.retrieval_mode = IMAGE_ONLY_RETRIEVAL
            image_only_job = build_training_job(
                image_only_args, siglip2_loader=_load_fake_siglip2
            )

            fused = fused_job.validate(1)
            image_only = image_only_job.validate(1)

        self.assertIn("image_only_mAP", fused.extras)
        self.assertIn("image_only_rank_1", fused.extras)
        self.assertNotIn("image_only_mAP", image_only.extras)

    def test_image_only_view_matches_image_only_retrieval_mode(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = _build_market_fixture(Path(tmp))
            job = build_training_job(
                _training_config(root), siglip2_loader=_load_fake_siglip2
            )
            job.model.eval()
            data = load_dataset_bundle(
                JobDataConfig("market1501", root),
                Siglip2ImageTransform(FakeSiglip2ImageProcessor()),
            )
            loader = torch.utils.data.DataLoader(
                data.gallery, batch_size=2, collate_fn=collate_reid_batches
            )

            both = _extract_features(
                job.model, loader, torch.device("cpu"), "fused", with_image_only=True
            )
            fused = _extract_features(job.model, loader, torch.device("cpu"), "fused")
            image_only = _extract_features(
                job.model, loader, torch.device("cpu"), IMAGE_ONLY_RETRIEVAL
            )

        self.assertTrue(torch.allclose(both.features, fused.features))
        self.assertTrue(torch.allclose(both.image_only.features, image_only.features))
        self.assertEqual(both.image_only.person_ids, fused.person_ids)

    def test_prcc_same_clothes_metrics_are_final_epoch_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = _build_prcc_fixture(Path(tmp))
            args = _training_config(root)
            args.dataset = "prcc"
            args.epochs = 2

            job = build_training_job(args, siglip2_loader=_load_fake_siglip2)
            intermediate = job.validate(1)
            final = job.validate(2)

        self.assertNotIn("same_clothes_mAP", intermediate.extras)
        self.assertIn("same_clothes_mAP", final.extras)

    def test_holdout_is_recorded_in_checkpoint_metadata_for_prcc_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            prcc_root = _build_prcc_fixture(Path(tmp) / "prcc", with_test=False)
            market_root = _build_market_fixture(Path(tmp) / "market")
            prcc_args = _training_config(prcc_root)
            prcc_args.dataset = "prcc"
            prcc_args.validation_holdout_ids = 1

            prcc_job = build_training_job(prcc_args, siglip2_loader=_load_fake_siglip2)
            market_job = build_training_job(
                _training_config(market_root), siglip2_loader=_load_fake_siglip2
            )

        self.assertEqual(prcc_job.checkpoint_metadata["validation_holdout_ids"], 1)
        self.assertEqual(prcc_job.checkpoint_metadata["validation_holdout_seed"], 0)
        self.assertEqual(prcc_job.checkpoint_metadata["num_train_ids"], 2)
        self.assertNotIn("validation_holdout_ids", market_job.checkpoint_metadata)

    def test_validation_reports_rerank_metrics_when_requested(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = _build_market_fixture(Path(tmp))
            args = _training_config(root)
            args.report_rerank = True

            job = build_training_job(args, siglip2_loader=_load_fake_siglip2)
            metrics = job.validate(1)

        self.assertIn(1, metrics.cmc)
        self.assertIn("rerank_mAP", metrics.extras)
        self.assertIn("rerank_rank_1", metrics.extras)

    def test_local_siglip2_checkpoint_is_loaded_explicitly(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = _build_market_fixture(Path(tmp))
            checkpoint = Path(tmp) / "siglip2_state.pth"
            model = FakeSiglip2(hidden_size=8, feature_dim=4)
            state = model.state_dict()
            key = "vision_model.embeddings.patch_embedding.weight"
            state[key] = torch.full_like(state[key], 0.25)
            torch.save(state, checkpoint)
            args = _training_config(root)
            args.siglip2_checkpoint = checkpoint

            job = build_training_job(args, siglip2_loader=_load_fake_siglip2)

        loaded = job.model.retrieval_model.image_encoder.siglip2_model.vision_model.embeddings.patch_embedding.weight
        self.assertTrue(torch.allclose(loaded, torch.full_like(loaded, 0.25)))

    def test_missing_siglip2_checkpoint_fails_at_startup(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = _build_market_fixture(Path(tmp))
            args = _training_config(root)
            args.siglip2_checkpoint = Path(tmp) / "missing.pth"

            with self.assertRaises(FileNotFoundError):
                build_training_job(args, siglip2_loader=_load_fake_siglip2)

    def test_siglip2_checkpoint_with_unexpected_keys_fails_at_startup(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = _build_market_fixture(Path(tmp))
            checkpoint = Path(tmp) / "bad_siglip2_state.pth"
            torch.save({"not_a_alignment_weight": torch.ones(1)}, checkpoint)
            args = _training_config(root)
            args.siglip2_checkpoint = checkpoint

            with self.assertRaisesRegex(
                ValueError, "unexpected SigLIP 2 checkpoint keys"
            ):
                build_training_job(args, siglip2_loader=_load_fake_siglip2)

    def test_no_freeze_image_encoder_stage2_reenables_encoder_after_stage1_freeze(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = _build_market_fixture(Path(tmp))
            args = _training_config(root)
            args.stage1_epochs = 1
            args.freeze_image_encoder_stage1 = True
            args.freeze_image_encoder_stage2 = False

            job = build_training_job(args, siglip2_loader=_load_fake_siglip2)

        siglip2_model = job.stage2.model.retrieval_model.image_encoder.siglip2_model
        self.assertGreater(_trainable_parameter_count(siglip2_model.vision_model), 0)

    def test_stage1_training_reapplies_stage1_freezing_before_epoch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = _build_market_fixture(Path(tmp))
            args = _training_config(root)
            args.stage1_epochs = 1
            args.freeze_image_encoder_stage1 = True
            args.freeze_image_encoder_stage2 = False
            job = build_training_job(args, siglip2_loader=_load_fake_siglip2)
            reporter = TrainBatchReporterRecorder()

            job.stage1.train_one_epoch(1, reporter)

        siglip2_model = job.stage1.model.retrieval_model.image_encoder.siglip2_model
        self.assertEqual(_trainable_parameter_count(siglip2_model.vision_model), 0)

    def test_stage1_learning_rate_follows_its_own_warmup_and_cosine(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = _build_market_fixture(Path(tmp))
            args = _training_config(root)
            args.stage1_epochs = 4
            args.stage1_lr_scheduler = "cosine"
            args.stage1_warmup_epochs = 2
            job = build_training_job(args, siglip2_loader=_load_fake_siglip2)
            reporter = TrainBatchReporterRecorder()

            observed = []
            for epoch in (1, 2, 3, 4):
                job.stage1.train_one_epoch(epoch, reporter)
                observed.append(job.stage1.optimizer.param_groups[0]["lr"])

        base = args.lr
        # Stage-1 epochs are 1-based, so epoch 1 is the first warmup step and
        # the cosine restarts from scale 1.0 on the first post-warmup epoch.
        self.assertAlmostEqual(observed[0], base * 0.5)
        self.assertAlmostEqual(observed[1], base)
        self.assertAlmostEqual(observed[2], base)
        self.assertAlmostEqual(observed[3], base * 0.5)

    def test_stage1_scheduler_is_absent_when_not_requested(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = _build_market_fixture(Path(tmp))
            args = _training_config(root)
            args.stage1_epochs = 2
            args.stage1_lr_scheduler = "none"
            job = build_training_job(args, siglip2_loader=_load_fake_siglip2)
            reporter = TrainBatchReporterRecorder()

            job.stage1.train_one_epoch(1, reporter)
            first = job.stage1.optimizer.param_groups[0]["lr"]
            job.stage1.train_one_epoch(2, reporter)
            second = job.stage1.optimizer.param_groups[0]["lr"]

        self.assertEqual(first, args.lr)
        self.assertEqual(second, args.lr)

    def test_no_freeze_image_encoder_stage2_works_without_stage1_epochs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = _build_market_fixture(Path(tmp))
            args = _training_config(root)
            args.stage1_epochs = 0
            args.freeze_image_encoder_stage1 = True
            args.freeze_image_encoder_stage2 = False

            job = build_training_job(args, siglip2_loader=_load_fake_siglip2)

        siglip2_model = job.model.retrieval_model.image_encoder.siglip2_model
        self.assertGreater(_trainable_parameter_count(siglip2_model.vision_model), 0)

    def test_stage2_can_freeze_prompt_bank_after_stage1(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = _build_market_fixture(Path(tmp))
            args = _training_config(root)
            args.stage1_epochs = 1
            args.freeze_prompt_bank_stage2 = True

            job = build_training_job(args, siglip2_loader=_load_fake_siglip2)
            job.stage2.train_one_epoch(2, TrainBatchReporterRecorder())

        prompt_bank = job.stage2.model.retrieval_model.prompt_bank
        self.assertEqual(_trainable_parameter_count(prompt_bank), 0)

    def test_stage2_frozen_prompts_encode_text_only_in_first_epoch(self):
        # With the prompt bank frozen in Stage-2 and the text encoder frozen,
        # the first epoch encodes the identity anchors (1 chunk: 2 train ids)
        # plus the per-camera retrieval text cache (1 call) and one exact
        # training-prompt TFC teacher call per micro-batch. Later epochs retain
        # only the TFC teacher call.
        with tempfile.TemporaryDirectory() as tmp:
            root = _build_market_fixture(Path(tmp))
            args = _training_config(root)
            args.freeze_prompt_bank_stage2 = True
            args.freeze_text_encoder = True
            args.epochs = 2

            job = build_training_job(args, siglip2_loader=_load_fake_siglip2)
            encoder = (
                job.model.retrieval_model.image_encoder.siglip2_model.text_model.encoder
            )
            encoder.call_count = 0
            job.train_one_epoch(1, TrainBatchReporterRecorder())
            first_epoch_calls = encoder.call_count
            job.train_one_epoch(2, TrainBatchReporterRecorder())
            second_epoch_calls = encoder.call_count - first_epoch_calls

        self.assertEqual(first_epoch_calls, 3)
        self.assertEqual(second_epoch_calls, 1)

    def test_stage2_frozen_prompts_with_trainable_text_encoder_recompute_anchors_each_epoch(
        self,
    ):
        # The anchors pass through the text encoder too: with the prompt bank
        # frozen but the text tower still training, the anchor matrix must be
        # re-encoded at every Stage-2 epoch start (and no camera cache may
        # exist), exactly like the fully unfrozen case.
        with tempfile.TemporaryDirectory() as tmp:
            root = _build_market_fixture(Path(tmp))
            args = _training_config(root)
            args.freeze_prompt_bank_stage2 = True
            args.freeze_text_encoder = False
            args.epochs = 2

            job = build_training_job(args, siglip2_loader=_load_fake_siglip2)
            encoder = (
                job.model.retrieval_model.image_encoder.siglip2_model.text_model.encoder
            )
            encoder.call_count = 0
            job.train_one_epoch(1, TrainBatchReporterRecorder())
            first_epoch_calls = encoder.call_count
            job.train_one_epoch(2, TrainBatchReporterRecorder())
            second_epoch_calls = encoder.call_count - first_epoch_calls

        self.assertIsNone(job.model.retrieval_model.inference_text_cache)
        self.assertEqual(first_epoch_calls, 3)
        self.assertEqual(second_epoch_calls, 3)

    def test_stage2_unfrozen_prompts_recompute_anchors_each_epoch(self):
        # Unfrozen prompt bank: the anchors act as a slowly-moving teacher and
        # are re-encoded at each Stage-2 epoch start (1 chunk) on top of the
        # per-batch retrieval text (1 batch in this fixture); no camera cache.
        with tempfile.TemporaryDirectory() as tmp:
            root = _build_market_fixture(Path(tmp))
            args = _training_config(root)
            args.epochs = 2

            job = build_training_job(args, siglip2_loader=_load_fake_siglip2)
            encoder = (
                job.model.retrieval_model.image_encoder.siglip2_model.text_model.encoder
            )
            encoder.call_count = 0
            job.train_one_epoch(1, TrainBatchReporterRecorder())
            first_epoch_calls = encoder.call_count
            job.train_one_epoch(2, TrainBatchReporterRecorder())
            second_epoch_calls = encoder.call_count - first_epoch_calls

        self.assertIsNone(job.model.retrieval_model.inference_text_cache)
        self.assertEqual(first_epoch_calls, 3)
        self.assertEqual(second_epoch_calls, 3)

    def test_failed_optimizer_window_rolls_back_tfc_center_updates(self):
        from t2c_reid.precision import PrecisionController

        with tempfile.TemporaryDirectory() as tmp:
            root = _build_market_fixture(Path(tmp))
            job = build_training_job(
                _training_config(root), siglip2_loader=_load_fake_siglip2
            )
            with mock.patch.object(PrecisionController, "step", return_value=False):
                job.train_one_epoch(1, TrainBatchReporterRecorder())

        self.assertFalse(bool(job.model.tfc_bank.visual_local_initialized.any()))
        self.assertFalse(bool(job.model.tfc_bank.text_local_initialized.any()))
        self.assertFalse(bool(job.model.tfc_bank.visual_global_initialized.any()))
        self.assertFalse(bool(job.model.tfc_bank.text_global_initialized.any()))

    def test_tfc_weight_zero_skips_training_prompt_teacher_and_center_updates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = _build_market_fixture(Path(tmp))
            args = _training_config(root)
            args.freeze_prompt_bank_stage2 = True
            args.freeze_text_encoder = True
            args.tfc_weight = 0.0
            args.epochs = 2

            job = build_training_job(args, siglip2_loader=_load_fake_siglip2)
            encoder = (
                job.model.retrieval_model.image_encoder.siglip2_model.text_model.encoder
            )
            encoder.call_count = 0
            job.train_one_epoch(1, TrainBatchReporterRecorder())
            first_epoch_calls = encoder.call_count
            job.train_one_epoch(2, TrainBatchReporterRecorder())
            second_epoch_calls = encoder.call_count - first_epoch_calls

        self.assertEqual(first_epoch_calls, 2)
        self.assertEqual(second_epoch_calls, 0)
        self.assertFalse(bool(job.model.tfc_bank.visual_local_initialized.any()))

    def test_stage2_camera_text_cache_matches_online_encoding(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = _build_market_fixture(Path(tmp))
            args = _training_config(root)
            args.freeze_prompt_bank_stage2 = True
            args.freeze_text_encoder = True

            job = build_training_job(args, siglip2_loader=_load_fake_siglip2)
            job.train_one_epoch(1, TrainBatchReporterRecorder())

            retrieval = job.model.retrieval_model
            cache = retrieval.inference_text_cache
            self.assertIsNotNone(cache)
            retrieval.set_inference_text_cache(None)
            with torch.no_grad():
                online = retrieval.encode_inference_text(torch.arange(cache.shape[0]))

        self.assertTrue(torch.equal(cache, online))

    def test_stage2_prompt_bank_remains_trainable_by_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = _build_market_fixture(Path(tmp))
            args = _training_config(root)

            job = build_training_job(args, siglip2_loader=_load_fake_siglip2)

        prompt_bank = job.model.retrieval_model.prompt_bank
        self.assertGreater(_trainable_parameter_count(prompt_bank), 0)

    def test_bnneck_adds_trainable_batch_norm_head(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = _build_market_fixture(Path(tmp))
            args = _training_config(root)
            args.reid_head = "bnneck"

            job = build_training_job(args, siglip2_loader=_load_fake_siglip2)

        # The head lives inside the retrieval model so training and eval share one path.
        self.assertFalse(hasattr(job.model, "feature_head"))
        self.assertGreater(
            _trainable_parameter_count(job.model.retrieval_model.feature_head), 0
        )

    def test_bnneck_keeps_batch_norm_bias_frozen_in_stage2(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = _build_market_fixture(Path(tmp))
            args = _training_config(root)
            args.reid_head = "bnneck"

            job = build_training_job(args, siglip2_loader=_load_fake_siglip2)

        self.assertFalse(job.model.retrieval_model.feature_head.bn.bias.requires_grad)

    def test_classifier_has_no_bias(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = _build_market_fixture(Path(tmp))

            job = build_training_job(
                _training_config(root), siglip2_loader=_load_fake_siglip2
            )

        self.assertIsNone(job.model.classifier.bias)

    def test_bnneck_head_params_land_in_new_optimizer_group(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = _build_market_fixture(Path(tmp))
            args = _training_config(root)
            args.reid_head = "bnneck"

            job = build_training_job(args, siglip2_loader=_load_fake_siglip2)

        _, new_group = _lookup_param_groups(job.optimizer)
        new_names = [
            _element_name(model=job.model, parameter=parameter)
            for parameter in new_group["params"]
        ]
        self.assertTrue(
            any("feature_head" in name for name in new_names),
            f"feature_head params missing from the 'new' group: {new_names}",
        )

    def test_siglip2_reid_model_encode_retrieval_delegates_to_retrieval_model(self):
        # The feature head now lives inside the retrieval model, so the wrapper
        # must return the retrieval model's feature unchanged (no second head).
        class StubRetrieval(torch.nn.Module):
            def encode_retrieval(self, images, camera_ids, retrieval_mode="fused"):
                return torch.ones(images.shape[0], 4)

        model = Siglip2ReIDTrainingModel(
            retrieval_model=StubRetrieval(),
            classifier=torch.nn.Linear(4, 2),
            tfc_bank=torch.nn.Module(),
        )

        output = model.encode_retrieval(
            torch.zeros(3, 3, 2, 2), torch.zeros(3, dtype=torch.long)
        )

        self.assertTrue(torch.equal(output, torch.ones(3, 4)))

    def test_validation_extracts_features_through_bnneck_head(self):
        # The built validation path must route retrieval through the BNNeck head
        # inside the retrieval model, so extracted features change when the head
        # applies a non-uniform per-dimension transform.
        with tempfile.TemporaryDirectory() as tmp:
            root = _build_market_fixture(Path(tmp))
            args = _training_config(root)
            args.reid_head = "bnneck"

            job = build_training_job(args, siglip2_loader=_load_fake_siglip2)

        head = job.model.retrieval_model.feature_head
        with torch.no_grad():
            head.bn.weight.copy_(torch.tensor([1.0, 2.0, 3.0, 4.0]))
            head.bn.running_mean.copy_(torch.tensor([0.0, 0.5, -0.5, 1.0]))
            head.bn.running_var.copy_(torch.tensor([1.0, 4.0, 0.25, 9.0]))
        job.model.eval()

        batch = ReIDImageBatch(
            images=torch.ones(2, 3, 14, 14),
            person_ids=torch.tensor([0, 1]),
            camera_ids=torch.tensor([0, 0]),
            original_person_ids=(10, 20),
            original_camera_ids=(1, 1),
        )
        device = torch.device("cpu")

        with_head = _extract_features(job.model, [batch], device, "fused")
        job.model.retrieval_model.feature_head = torch.nn.Identity()
        without_head = _extract_features(job.model, [batch], device, "fused")

        self.assertFalse(torch.allclose(with_head.features, without_head.features))

    def test_stage2_training_step_runs_single_image_forward_per_batch(self):
        # TFC center updates must reuse the loss forward's features instead of
        # running a second full no-grad forward per batch.
        with tempfile.TemporaryDirectory() as tmp:
            root = _build_market_fixture(Path(tmp))
            job = build_training_job(
                _training_config(root), siglip2_loader=_load_fake_siglip2
            )
            siglip2 = job.model.retrieval_model.image_encoder.siglip2_model
            siglip2.image_feature_calls = 0

            job.train_one_epoch(1, TrainBatchReporterRecorder())

        self.assertEqual(siglip2.image_feature_calls, 1)

    def test_persistent_workers_require_positive_worker_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = _build_market_fixture(Path(tmp))
            args = _training_config(root)
            args.persistent_workers = True

            with self.assertRaisesRegex(ValueError, "persistent_workers"):
                build_training_job(args, siglip2_loader=_load_fake_siglip2)

    def test_hydra_default_keeps_stage2_image_encoder_unfrozen(self):
        from t2c_reid.jobs.siglip2_reid import _job_config_from_training_config

        training_config = compose_training_config()
        job_config = _job_config_from_training_config(training_config)

        self.assertFalse(job_config.freeze_image_encoder_stage2)

    def test_image_encoder_lr_creates_separate_param_group_for_unfrozen_stage2(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = _build_market_fixture(Path(tmp))
            args = _training_config(root)
            args.stage1_epochs = 0
            args.freeze_image_encoder_stage1 = True
            args.freeze_image_encoder_stage2 = False  # default unfrozen
            args.lr = 0.001
            args.image_encoder_lr = 5e-5

            job = build_training_job(args, siglip2_loader=_load_fake_siglip2)

        optimizer = job.optimizer
        backbone_group, new_group = _lookup_param_groups(optimizer)
        self.assertAlmostEqual(backbone_group["lr"], 5e-5)
        self.assertAlmostEqual(new_group["lr"], 0.001)
        # Filtered by name: backbone group should hold only visual_projection params.
        backbone_names = [
            _element_name(model=job.model, parameter=parameter)
            for parameter in backbone_group["params"]
        ]
        self.assertTrue(
            all(
                "visual_projection" in name or "vision_model" in name
                for name in backbone_names
            ),
            f"backbone group contains non-backbone params: {backbone_names}",
        )

    def test_beta_schedule_ramps_from_zero_to_beta_over_warmup(self):
        schedule = BetaSchedule(beta=0.1, warmup_epochs=5)
        self.assertAlmostEqual(schedule.effective_beta(1), 0.0)
        self.assertAlmostEqual(schedule.effective_beta(2), 0.02)
        self.assertAlmostEqual(schedule.effective_beta(5), 0.08)
        self.assertAlmostEqual(schedule.effective_beta(6), 0.1)
        self.assertAlmostEqual(schedule.effective_beta(120), 0.1)

    def test_beta_schedule_uses_stage_local_epoch_offset(self):
        schedule = BetaSchedule(beta=0.1, warmup_epochs=5)

        self.assertAlmostEqual(schedule.effective_beta(stage_epoch=1), 0.0)
        self.assertAlmostEqual(schedule.effective_beta(stage_epoch=2), 0.02)
        self.assertAlmostEqual(schedule.effective_beta(stage_epoch=5), 0.08)
        self.assertAlmostEqual(schedule.effective_beta(stage_epoch=6), 0.1)

    def test_beta_schedule_zero_warmup_returns_constant_beta(self):
        schedule = BetaSchedule(beta=0.1, warmup_epochs=0)
        self.assertAlmostEqual(schedule.effective_beta(1), 0.1)
        self.assertAlmostEqual(schedule.effective_beta(10), 0.1)

    def test_beta_schedule_applies_to_model_retrieval_beta(self):
        class Siglip2ReIDTrainingModelStub(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.retrieval_model = torch.nn.Module()
                self.retrieval_model.beta = 999.0

        stub = Siglip2ReIDTrainingModelStub()
        warmup_schedule = BetaSchedule(beta=0.3, warmup_epochs=2)
        warmup_schedule.apply(stub, epoch=1)
        self.assertEqual(stub.retrieval_model.beta, 0.0)
        warmup_schedule.apply(stub, epoch=3)
        self.assertAlmostEqual(stub.retrieval_model.beta, 0.3)

    def test_beta_schedule_apply_uses_stage_first_epoch(self):
        class Siglip2ReIDTrainingModelStub(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.retrieval_model = torch.nn.Module()
                self.retrieval_model.beta = 999.0

        stub = Siglip2ReIDTrainingModelStub()
        schedule = BetaSchedule(beta=0.1, warmup_epochs=5, first_epoch=11)

        schedule.apply(stub, epoch=11)
        self.assertAlmostEqual(stub.retrieval_model.beta, 0.0)

        schedule.apply(stub, epoch=15)
        self.assertAlmostEqual(stub.retrieval_model.beta, 0.08)

        schedule.apply(stub, epoch=16)
        self.assertAlmostEqual(stub.retrieval_model.beta, 0.1)

    def test_stage_lr_scheduler_warmup_then_cosine(self):
        scheduler = StageLRScheduler(base_lrs=(1.0,), total_epochs=10, warmup_epochs=2)

        self.assertAlmostEqual(scheduler.scale(1), 0.5)
        self.assertAlmostEqual(scheduler.scale(2), 1.0)
        self.assertAlmostEqual(scheduler.scale(3), 1.0)
        self.assertLess(scheduler.scale(10), 0.05)
        self.assertGreaterEqual(scheduler.scale(10), 0.0)

    def test_stage_lr_scheduler_apply_scales_groups_by_stage_epoch(self):
        parameter = torch.nn.Parameter(torch.zeros(1))
        optimizer = torch.optim.AdamW(
            [{"params": [parameter], "lr": 1e-4, "name": "new"}]
        )
        scheduler = StageLRScheduler(
            base_lrs=(1e-4,), total_epochs=10, warmup_epochs=2, first_epoch=11
        )

        scheduler.apply(optimizer, epoch=11)  # stage epoch 1 -> 0.5x warmup

        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 0.5e-4)

    def test_stage2_cosine_scheduler_changes_lr_across_epochs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = _build_market_fixture(Path(tmp))
            args = _training_config(root)
            args.stage2_lr_scheduler = "cosine"
            args.stage2_warmup_epochs = 2
            args.epochs = 10

            job = build_training_job(args, siglip2_loader=_load_fake_siglip2)
            first = job.train_one_epoch(1, TrainBatchReporterRecorder())["lr"]
            second = job.train_one_epoch(2, TrainBatchReporterRecorder())["lr"]

        self.assertNotEqual(first, second)

    def test_stage2_scheduler_none_keeps_lr_constant(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = _build_market_fixture(Path(tmp))
            args = _training_config(root)
            args.stage2_lr_scheduler = "none"
            args.epochs = 10

            job = build_training_job(args, siglip2_loader=_load_fake_siglip2)
            first = job.train_one_epoch(1, TrainBatchReporterRecorder())["lr"]
            second = job.train_one_epoch(2, TrainBatchReporterRecorder())["lr"]

        self.assertEqual(first, second)

    def test_job_config_reads_num_instances(self):
        from t2c_reid.jobs.siglip2_reid import _job_config_from_training_config

        args = _training_config(Path("."))
        args.num_instances = 4

        config = _job_config_from_training_config(args)

        self.assertEqual(config.num_instances, 4)

    def test_job_config_reads_id_logit_scale(self):
        from t2c_reid.jobs.siglip2_reid import _job_config_from_training_config

        args = _training_config(Path("."))
        args.id_logit_scale = 10.0

        config = _job_config_from_training_config(args)

        self.assertEqual(config.id_logit_scale, 10.0)

    def test_job_config_reads_triplet_metric(self):
        from t2c_reid.jobs.siglip2_reid import _job_config_from_training_config

        args = _training_config(Path("."))
        args.triplet_metric = "cosine"

        config = _job_config_from_training_config(args)

        self.assertEqual(config.triplet_metric, "cosine")

    def test_job_config_uses_hydra_default_triplet_metric(self):
        from t2c_reid.jobs.siglip2_reid import _job_config_from_training_config

        training_config = compose_training_config()

        config = _job_config_from_training_config(training_config)

        self.assertEqual(config.triplet_metric, "euclidean")

    def test_stage_metadata_includes_triplet_metric(self):

        args = _training_config(Path("."))
        args.triplet_metric = "cosine"

        metadata = _stage_metadata_for(args)

        self.assertEqual(metadata.get("triplet_metric"), "cosine")

    def test_job_config_reads_fixture_num_instances(self):
        from t2c_reid.jobs.siglip2_reid import _job_config_from_training_config

        args = _training_config(Path("."))

        config = _job_config_from_training_config(args)

        self.assertEqual(config.num_instances, 2)

    def test_train_loader_uses_configured_num_instances(self):
        from t2c_reid.jobs.siglip2_reid import (
            _job_config_from_training_config,
            _train_loader,
        )

        class _StubDataset(torch.utils.data.Dataset):
            def __init__(self, person_ids):
                self._person_ids = tuple(person_ids)

            @property
            def person_ids(self):
                return self._person_ids

            def __len__(self):
                return len(self._person_ids)

            def __getitem__(self, index):
                return index

        args = _training_config(Path("."))
        args.num_instances = 4
        args.batch_size = 8
        config = _job_config_from_training_config(args)
        dataset = _StubDataset([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3])

        loader = _train_loader(dataset, config)

        self.assertEqual(loader.batch_sampler._instances_per_identity, 4)
        self.assertEqual(loader.batch_sampler._identities_per_batch, 2)

    def test_stage_loss_configs_read_logit_scale_from_siglip2_model(self):
        import math

        from t2c_reid.jobs.siglip2_reid import (
            _job_config_from_training_config,
            _stage1_loss_config,
            _stage2_loss_config,
        )

        siglip2 = FakeSiglip2(
            hidden_size=8, feature_dim=4
        )  # logit_scale parameter = 1.0

        stage1 = _stage1_loss_config(siglip2)
        self.assertAlmostEqual(stage1.logit_scale, math.e, places=5)
        self.assertEqual(stage1.logit_bias, -1.0)

        config = _job_config_from_training_config(_training_config(Path(".")))
        stage2 = _stage2_loss_config(config, siglip2)
        self.assertAlmostEqual(stage2.logit_scale, math.e, places=5)
        self.assertEqual(stage2.logit_bias, -1.0)
        self.assertEqual(stage2.triplet_margin, config.triplet_margin)
        self.assertEqual(stage2.triplet_metric, config.triplet_metric)
        self.assertEqual(stage2.id_logit_scale, config.id_logit_scale)
        self.assertEqual(stage2.alignment_weight, config.alignment_weight)

    def test_siglip_logit_scale_and_bias_are_frozen_and_not_optimized(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = _build_market_fixture(Path(tmp))

            job = build_training_job(
                _training_config(root), siglip2_loader=_load_fake_siglip2
            )

        siglip2 = job.model.retrieval_model.image_encoder.siglip2_model
        self.assertFalse(siglip2.logit_scale.requires_grad)
        self.assertFalse(siglip2.logit_bias.requires_grad)
        optimized = {
            id(parameter)
            for group in job.optimizer.param_groups
            for parameter in group["params"]
        }
        self.assertNotIn(id(siglip2.logit_scale), optimized)
        self.assertNotIn(id(siglip2.logit_bias), optimized)

    def test_build_training_job_wraps_prompts_in_natural_language_template(self):
        # The learnable slots must be framed as
        # "a photo of a <slots> person ." with tokenizer-encoded template ids
        # (SOT/EOS stripped from the encoded fragments).
        with tempfile.TemporaryDirectory() as tmp:
            root = _build_market_fixture(Path(tmp))

            job = build_training_job(
                _training_config(root), siglip2_loader=_load_fake_siglip2
            )

        text_encoder = job.model.retrieval_model.text_encoder
        self.assertEqual(text_encoder.prefix_token_ids, (5, 6, 7, 5))
        self.assertEqual(text_encoder.suffix_token_ids, (8, 9))

    def test_build_training_job_requires_tokenizer_for_prompt_template(self):
        def load_without_tokenizer(model_name: str) -> Siglip2LoadResult:
            return Siglip2LoadResult(
                FakeSiglip2(hidden_size=8, feature_dim=4),
                FakeSiglip2ImageProcessor(),
                tokenizer=None,
            )

        with tempfile.TemporaryDirectory() as tmp:
            root = _build_market_fixture(Path(tmp))

            with self.assertRaisesRegex(ValueError, "tokenizer"):
                build_training_job(
                    _training_config(root), siglip2_loader=load_without_tokenizer
                )

    def test_job_config_reads_sie_coe(self):
        from t2c_reid.jobs.siglip2_reid import _job_config_from_training_config

        args = _training_config(Path("."))
        args.sie_coe = 1.5

        config = _job_config_from_training_config(args)

        self.assertEqual(config.sie_coe, 1.5)

    def test_job_config_sie_coe_defaults_to_zero_when_absent(self):
        from t2c_reid.jobs.siglip2_reid import _job_config_from_training_config

        config = _job_config_from_training_config(_training_config(Path(".")))

        self.assertEqual(config.sie_coe, 0.0)

    def test_stage_metadata_includes_sie_coe(self):

        args = _training_config(Path("."))
        args.sie_coe = 1.5

        metadata = _stage_metadata_for(args)

        self.assertEqual(metadata.get("sie_coe"), 1.5)

    def test_sie_disabled_by_default_creates_no_embedding(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = _build_market_fixture(Path(tmp))

            job = build_training_job(
                _training_config(root), siglip2_loader=_load_fake_siglip2
            )

        self.assertIsNone(job.model.retrieval_model.image_encoder.sie_embedding)

    def test_sie_embedding_sized_by_dataset_cameras_when_enabled(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = _build_market_fixture(Path(tmp))  # two cameras: c1, c2
            args = _training_config(root)
            args.sie_coe = 1.0

            job = build_training_job(args, siglip2_loader=_load_fake_siglip2)

        sie = job.model.retrieval_model.image_encoder.sie_embedding
        self.assertIsNotNone(sie)
        self.assertEqual(sie.num_embeddings, 2)

    def test_sie_params_land_in_new_optimizer_group(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = _build_market_fixture(Path(tmp))
            args = _training_config(root)
            args.sie_coe = 1.0

            job = build_training_job(args, siglip2_loader=_load_fake_siglip2)

        backbone_group, new_group = _lookup_param_groups(job.optimizer)
        backbone_names = [
            _element_name(model=job.model, parameter=p)
            for p in backbone_group["params"]
        ]
        new_names = [
            _element_name(model=job.model, parameter=p) for p in new_group["params"]
        ]
        self.assertTrue(
            any("sie_embedding" in name for name in new_names),
            f"sie_embedding params missing from the 'new' group: {new_names}",
        )
        self.assertFalse(any("sie_embedding" in name for name in backbone_names))

    def test_sie_embedding_follows_image_encoder_freeze_per_stage(self):
        from t2c_reid.jobs.siglip2_reid import (
            _apply_freezing,
            _job_config_from_training_config,
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = _build_market_fixture(Path(tmp))
            args = _training_config(root)
            args.sie_coe = 1.0
            args.stage1_epochs = 1
            args.freeze_image_encoder_stage1 = True
            args.freeze_image_encoder_stage2 = False

            job = build_training_job(args, siglip2_loader=_load_fake_siglip2)
            config = _job_config_from_training_config(args)
            model = job.stage2.model
            sie = model.retrieval_model.image_encoder.sie_embedding

            _apply_freezing(model, config, "stage1")
            self.assertFalse(sie.weight.requires_grad)

            _apply_freezing(model, config, "stage2")
            self.assertTrue(sie.weight.requires_grad)

    def test_job_config_stage1_feature_cache_defaults_true(self):
        from t2c_reid.jobs.siglip2_reid import _job_config_from_training_config

        config = _job_config_from_training_config(_training_config(Path(".")))

        self.assertTrue(config.stage1_feature_cache)

    def test_stage_metadata_includes_stage1_feature_cache(self):

        metadata = _stage_metadata_for(_training_config(Path(".")))

        self.assertIs(metadata.get("stage1_feature_cache"), True)

    def test_stage_metadata_includes_image_size_and_prompt_template(self):

        metadata = _stage_metadata_for(_training_config(Path(".")))

        self.assertEqual(metadata.get("image_size"), "392x196")
        self.assertEqual(metadata.get("checkpoint_schema_version"), 3)
        self.assertEqual(metadata.get("tfc_version"), "camera_aware_cross_modal_v1")
        self.assertEqual(metadata.get("tfc_tail_momentum"), 0.9)
        self.assertEqual(metadata.get("tfc_class_balance_beta"), 0.9999)
        self.assertEqual(metadata.get("prompt_template_prefix"), "a photo of a")
        self.assertEqual(metadata.get("prompt_template_suffix"), "person .")

    def test_stage1_optimizer_excludes_camera_transfer_logits(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = _build_market_fixture(Path(tmp))
            args = _training_config(root)
            args.stage1_epochs = 1
            job = build_training_job(args, siglip2_loader=_load_fake_siglip2)

        transfer = job.stage1.model.tfc_bank.camera_transfer_logits
        stage1_parameters = {
            id(parameter)
            for group in job.stage1.optimizer.param_groups
            for parameter in group["params"]
        }
        stage2_parameters = {
            id(parameter)
            for group in job.stage2.optimizer.param_groups
            for parameter in group["params"]
        }
        self.assertNotIn(id(transfer), stage1_parameters)
        self.assertIn(id(transfer), stage2_parameters)

    def test_tfc_config_rejects_invalid_momentum_and_empty_main_loss(self):
        from t2c_reid.jobs.siglip2_reid import _job_config_from_training_config

        args = _training_config(Path("."))
        args.tfc_tail_momentum = 0.4
        with self.assertRaisesRegex(ValueError, "head <= tail"):
            _job_config_from_training_config(args)

        args = _training_config(Path("."))
        args.tfc_local_weight = 0.0
        args.tfc_global_weight = 0.0
        args.tfc_cross_modal_weight = 0.0
        args.tfc_cross_camera_weight = 0.0
        with self.assertRaisesRegex(ValueError, "at least one positive"):
            _job_config_from_training_config(args)

    def test_optimizer_uses_no_decay_groups_for_norms_bias_prompts_and_sie(self):
        #  SigLIP 2-ReID-style AdamW grouping: 1-D parameters (norm weights/biases),
        # the prompt bank, and the SIE embedding train without weight decay;
        # every other weight uses 1e-4.
        with tempfile.TemporaryDirectory() as tmp:
            root = _build_market_fixture(Path(tmp))
            args = _training_config(root)
            args.reid_head = "bnneck"
            args.sie_coe = 3.0

            job = build_training_job(args, siglip2_loader=_load_fake_siglip2)

        parameters_by_name = dict(job.model.named_parameters())
        seen_no_decay = 0
        seen_decay = 0
        for group in job.optimizer.param_groups:
            for parameter in group["params"]:
                name = _element_name(model=job.model, parameter=parameter)
                expects_no_decay = (
                    parameters_by_name[name].ndim <= 1
                    or name.startswith(
                        (
                            "retrieval_model.prompt_bank.",
                            "retrieval_model.image_encoder.sie_embedding.",
                        )
                    )
                    or name == "tfc_bank.camera_transfer_logits"
                )
                if expects_no_decay:
                    self.assertEqual(float(group["weight_decay"]), 0.0, name)
                    seen_no_decay += 1
                else:
                    self.assertEqual(float(group["weight_decay"]), 1e-4, name)
                    seen_decay += 1
        self.assertGreater(seen_no_decay, 0)
        self.assertGreater(seen_decay, 0)

    def test_stage1_feature_cache_rejects_trainable_stage1_image_encoder(self):
        # Cached image features are only valid while the Stage-1 image tower
        # is frozen; the conflicting flag combination must fail at build time.
        args = _training_config(Path("."))
        args.freeze_image_encoder_stage1 = False
        args.stage1_feature_cache = True

        with self.assertRaisesRegex(
            ValueError, r"stage1_feature_cache.*freeze_image_encoder_stage1"
        ):
            build_training_job(args, siglip2_loader=_load_fake_siglip2)

    def test_stage1_feature_cache_can_be_disabled_with_trainable_image_encoder(self):
        from t2c_reid.jobs.siglip2_reid import _job_config_from_training_config

        args = _training_config(Path("."))
        args.freeze_image_encoder_stage1 = False
        args.stage1_feature_cache = False

        config = _job_config_from_training_config(args)

        self.assertFalse(config.stage1_feature_cache)

    def test_load_dataset_bundle_builds_train_eval_view_with_eval_transform(self):
        class ConstantTransform:
            def __init__(self, value: float):
                self.value = value

            def __call__(self, image):
                return torch.full((3, image.height, image.width), self.value)

        with tempfile.TemporaryDirectory() as tmp:
            root = _build_market_fixture(Path(tmp))
            transforms = TransformBundle(
                train=ConstantTransform(2.0),
                eval=ConstantTransform(1.0),
            )

            data = load_dataset_bundle(JobDataConfig("market1501", root), transforms)

            self.assertEqual(len(data.train_eval), len(data.train))
            self.assertEqual(data.train_eval.person_ids, data.train.person_ids)
            self.assertTrue(torch.equal(data.train_eval[0].image, torch.ones(3, 2, 2)))

    def test_stage1_cache_runs_image_tower_only_in_first_epoch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = _build_market_fixture(Path(tmp))
            args = _training_config(root)
            args.stage1_epochs = 2

            job = build_training_job(args, siglip2_loader=_load_fake_siglip2)
            siglip2 = job.stage1.model.retrieval_model.image_encoder.siglip2_model
            siglip2.image_feature_calls = 0
            first_reporter = TrainBatchReporterRecorder()
            job.stage1.train_one_epoch(1, first_reporter)
            first_epoch_calls = siglip2.image_feature_calls
            second_reporter = TrainBatchReporterRecorder()
            job.stage1.train_one_epoch(2, second_reporter)
            second_epoch_calls = siglip2.image_feature_calls - first_epoch_calls

        # Extraction is one sequential pass (4 train images, batch 4 -> 1 call);
        # every training step afterwards must come from the cache.
        self.assertEqual(first_epoch_calls, 1)
        self.assertEqual(second_epoch_calls, 0)
        # The first epoch trains too, with the same number of steps per epoch.
        self.assertEqual(len(first_reporter.batch_reports), 1)
        self.assertEqual(len(second_reporter.batch_reports), 1)
        self.assertIn("loss", first_reporter.batch_reports[0])
        self.assertIn("lr", first_reporter.batch_reports[0])

    def test_stage1_cache_disabled_runs_image_tower_every_epoch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = _build_market_fixture(Path(tmp))
            args = _training_config(root)
            args.stage1_epochs = 2
            args.stage1_feature_cache = False

            job = build_training_job(args, siglip2_loader=_load_fake_siglip2)
            siglip2 = job.stage1.model.retrieval_model.image_encoder.siglip2_model
            siglip2.image_feature_calls = 0
            job.stage1.train_one_epoch(1, TrainBatchReporterRecorder())
            first_epoch_calls = siglip2.image_feature_calls
            job.stage1.train_one_epoch(2, TrainBatchReporterRecorder())
            second_epoch_calls = siglip2.image_feature_calls - first_epoch_calls

        self.assertEqual(first_epoch_calls, 1)
        self.assertEqual(second_epoch_calls, 1)

    def test_stage1_cache_extraction_uses_eval_transform(self):
        # The train transform is poisoned: with the cache enabled, Stage-1 must
        # never load images through it (extraction uses the eval transform and
        # later steps read the cache), so the epoch completes without raising.
        class PoisonTrainTransform:
            def __init__(self, image_processor, image_size=None):
                self.image_processor = image_processor
                self.image_size = image_size

            def __call__(self, image):
                raise AssertionError(
                    "train transform must not run while the stage1 feature cache is enabled"
                )

        with tempfile.TemporaryDirectory() as tmp:
            root = _build_market_fixture(Path(tmp))
            args = _training_config(root)
            args.stage1_epochs = 1
            args.data_backend = "python"
            with mock.patch(
                "t2c_reid.jobs.siglip2_reid.Siglip2TrainImageTransform",
                PoisonTrainTransform,
            ):
                job = build_training_job(args, siglip2_loader=_load_fake_siglip2)
                metrics = job.stage1.train_one_epoch(1, TrainBatchReporterRecorder())

        self.assertIn("loss", metrics)

    def test_stage1_cached_epoch_losses_match_online(self):
        # With a deterministic transform (train == eval) and seeded PK sampling,
        # the cached path must reproduce the online per-epoch losses exactly.
        def run_stage1(root: Path, feature_cache: bool) -> list[float]:
            random.seed(7)
            torch.manual_seed(7)
            args = _training_config(root)
            args.stage1_epochs = 2
            args.stage1_feature_cache = feature_cache
            with mock.patch(
                "t2c_reid.jobs.siglip2_reid.Siglip2TrainImageTransform",
                Siglip2ImageTransform,
            ):
                job = build_training_job(args, siglip2_loader=_load_fake_siglip2)
            losses = []
            for epoch in (1, 2):
                random.seed(100 + epoch)
                metrics = job.stage1.train_one_epoch(
                    epoch, TrainBatchReporterRecorder()
                )
                losses.append(metrics["loss"])
            return losses

        with tempfile.TemporaryDirectory() as tmp:
            root = _build_market_fixture(Path(tmp))
            cached = run_stage1(root, feature_cache=True)
            online = run_stage1(root, feature_cache=False)

        self.assertEqual(len(cached), 2)
        for cached_loss, online_loss in zip(cached, online):
            self.assertAlmostEqual(cached_loss, online_loss, places=10)

    def test_gradient_accumulation_reports_optimizer_windows(self):
        model, reporter, _ = _run_accumulation_fixture(
            targets=(0.0, 2.0, 4.0, 6.0), accumulation_steps=2
        )

        self.assertEqual(len(reporter.batch_reports), 2)
        self.assertNotEqual(float(model.weight.detach()), 0.0)

    def test_gradient_accumulation_tail_window_uses_actual_size(self):
        model, reporter, metrics = _run_accumulation_fixture(
            targets=(0.0, 2.0, 4.0), accumulation_steps=2
        )

        # Window 1 averages gradients for targets 0 and 2: w 0 -> 0.2.
        # Tail window has one item and must use its full gradient: 0.2 -> 0.96.
        self.assertAlmostEqual(float(model.weight.detach()), 0.96, places=6)
        self.assertEqual(len(reporter.batch_reports), 2)
        self.assertAlmostEqual(metrics["loss"], (0.0 + 4.0 + 14.44) / 3.0, places=6)


def _run_accumulation_fixture(
    *,
    targets: tuple[float, ...],
    accumulation_steps: int,
):
    from t2c_reid.jobs.siglip2_reid import (
        LoaderBundle,
        StageTrainingRuntime,
        _train_one_epoch,
    )
    from t2c_reid.precision import PrecisionController, PrecisionPolicy
    from t2c_reid.training import Stage1LossBreakdown, Stage1LossConfig

    model = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.zero_()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    runtime = StageTrainingRuntime(
        model=model,
        loaders=LoaderBundle(train=list(targets), query=[], gallery=[]),
        optimizer=optimizer,
        stage="stage1",
        loss_config=Stage1LossConfig(),
        device=torch.device("cpu"),
        precision=PrecisionController(PrecisionPolicy("fp32", "fp32", "cpu")),
        gradient_accumulation_steps=accumulation_steps,
    )
    reporter = TrainBatchReporterRecorder()

    def breakdown(_runtime, target, _stage):
        loss = (model.weight.squeeze() - float(target)).square()
        return Stage1LossBreakdown(alignment=loss)

    with mock.patch(
        "t2c_reid.jobs.siglip2_reid._micro_batch_breakdown",
        side_effect=breakdown,
    ):
        metrics = _train_one_epoch(runtime)(1, reporter)
    return model, reporter, metrics


def _load_fake_siglip2(model_name: str) -> Siglip2LoadResult:
    return Siglip2LoadResult(
        FakeSiglip2(hidden_size=8, feature_dim=4),
        FakeSiglip2ImageProcessor(),
        tokenizer=FakeSiglip2Tokenizer(),
    )


def _stage_metadata_for(training_config: TrainingConfig):
    from t2c_reid.jobs.siglip2_reid import (
        _job_config_from_training_config,
        _stage_metadata,
        _validate_loaded_siglip2,
    )

    config = _job_config_from_training_config(training_config)
    loaded = _load_fake_siglip2(config.siglip2_model_name)
    spec = _validate_loaded_siglip2(loaded, config.image_size)
    data = SimpleNamespace(
        num_train_ids=2,
        num_cameras=2,
        identity_camera_counts=torch.ones(2, 2, dtype=torch.long),
    )
    return _stage_metadata(config, spec, data)


def _trainable_parameter_count(module: torch.nn.Module) -> int:
    return sum(
        parameter.numel()
        for parameter in module.parameters()
        if parameter.requires_grad
    )


def _lookup_param_groups(optimizer: torch.optim.Optimizer) -> tuple[dict, dict]:
    """Return synthetic (backbone, new) param groups for the grouped-lr optimizer.

    The optimizer splits each lr family into a decay and a no-decay group;
    tests that only care about lr-family membership see the merged view.
    """
    merged: dict[str, dict] = {}
    for group in optimizer.param_groups:
        name = str(group.get("name", ""))
        family = (
            "backbone"
            if name.startswith("backbone")
            else "new"
            if name.startswith("new")
            else name
        )
        if family not in merged:
            merged[family] = {"params": [], "lr": group["lr"], "name": family}
        merged[family]["params"].extend(group["params"])
    for required in ("backbone", "new"):
        if required not in merged:
            raise AssertionError(
                f"optimizer is missing required param group {required!r}; "
                f"found names: {sorted(str(group.get('name', '')) for group in optimizer.param_groups)}"
            )
    return merged["backbone"], merged["new"]


def _element_name(model: torch.nn.Module, parameter: torch.nn.Parameter) -> str:
    for name, candidate in model.named_parameters():
        if candidate is parameter:
            return name
    raise AssertionError("parameter is not present on model.named_parameters()")


class TrainBatchReporterRecorder:
    def __init__(self):
        self.batch_reports: list[dict[str, float]] = []

    def batches(self, iterable):
        return iterable

    def report_batch(self, metrics):
        self.batch_reports.append(dict(metrics))


class RetrievalModeRecorder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.retrieval_modes: list[str] = []
        self.seen_images: list[torch.Tensor] = []

    def encode_retrieval(
        self,
        images: torch.Tensor,
        camera_ids: torch.Tensor,
        retrieval_mode: str = "fused",
    ) -> torch.Tensor:
        self.retrieval_modes.append(retrieval_mode)
        self.seen_images.append(images.clone())
        return torch.ones(images.shape[0], 4)


class DirectionalRetrievalStub(torch.nn.Module):
    """Returns a unit feature whose direction depends on which half is brighter.

    Lets a flip-TTA test observe that the two views were actually averaged
    rather than one of them being silently dropped.
    """

    def forward(self, images: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError

    def encode_retrieval(
        self,
        images: torch.Tensor,
        camera_ids: torch.Tensor,
        retrieval_mode: str = "fused",
    ) -> torch.Tensor:
        width = images.shape[3]
        left = images[:, :, :, : width // 2].mean(dim=(1, 2, 3))
        right = images[:, :, :, width // 2 :].mean(dim=(1, 2, 3))
        features = torch.stack([left, right], dim=1)
        return torch.nn.functional.normalize(features, dim=1)


def _asymmetric_batch() -> ReIDImageBatch:
    """A batch whose images differ left-to-right, so a mirror is observable."""

    images = torch.zeros(2, 3, 2, 4)
    images[:, :, :, :2] = 1.0
    return ReIDImageBatch(
        images=images,
        person_ids=torch.tensor([0, 1]),
        camera_ids=torch.tensor([1, 2]),
        original_person_ids=(10, 20),
        original_camera_ids=(1, 2),
    )


def _training_config(root: Path) -> TrainingConfig:
    return compose_training_config(
        [
            "dataset=market1501",
            f"data_root={root.as_posix()}",
            "batch_size=4",
            "eval_batch_size=16",
            "gradient_accumulation_steps=4",
            "num_instances=2",
            "num_workers=0",
            "rust_data_threads=1",
            "lr=0.001",
            "image_encoder_lr=0.00005",
            "device=cpu",
            "context_length=2",
            "label_smoothing=0.0",
            "stage1_epochs=0",
            "epochs=1",
            "validation_interval=1",
            "stage1_lr_scheduler=none",
            "stage1_warmup_epochs=0",
            "stage2_lr_scheduler=none",
            "stage2_warmup_epochs=0",
            "reid_head=linear",
            "grad_clip_norm=0.0",
        ]
    )


def _build_market_fixture(root: Path) -> Path:
    _write_market_image(root / "bounding_box_train" / "0001_c1s1_000001_01.jpg", "red")
    _write_market_image(root / "bounding_box_train" / "0001_c2s1_000002_01.jpg", "red")
    _write_market_image(root / "bounding_box_train" / "0002_c1s1_000003_01.jpg", "blue")
    _write_market_image(root / "bounding_box_train" / "0002_c2s1_000004_01.jpg", "blue")
    _write_market_image(root / "query" / "0003_c1s1_000004_01.jpg", "green")
    _write_market_image(root / "bounding_box_test" / "0003_c2s1_000005_01.jpg", "green")
    _write_market_image(root / "bounding_box_test" / "0004_c1s1_000006_01.jpg", "blue")
    return root


def _build_prcc_fixture(root: Path, *, with_test: bool = True) -> Path:
    train = root / "rgb" / "train"
    for pid, color in ((1, "red"), (2, "blue"), (3, "yellow")):
        for camera in "ABC":
            for index in range(2):
                _write_market_image(
                    train / f"{pid:03d}" / f"{camera}_cropped_rgb{index:03d}.jpg", color
                )
    if with_test:
        test = root / "rgb" / "test"
        _write_market_image(test / "A" / "004" / "cropped_rgb001.jpg", "green")
        _write_market_image(test / "A" / "005" / "cropped_rgb002.jpg", "white")
        _write_market_image(test / "B" / "004" / "cropped_rgb003.jpg", "green")
        _write_market_image(test / "C" / "004" / "cropped_rgb004.jpg", "black")
    return root


def _build_market_fixture_without_positive_pairs(root: Path) -> Path:
    _write_market_image(root / "bounding_box_train" / "0001_c1s1_000001_01.jpg", "red")
    _write_market_image(root / "bounding_box_train" / "0002_c1s1_000002_01.jpg", "blue")
    _write_market_image(root / "query" / "0003_c1s1_000003_01.jpg", "green")
    _write_market_image(root / "bounding_box_test" / "0003_c2s1_000004_01.jpg", "green")
    return root


def _write_market_image(path: Path, color: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (2, 2), color=color).save(path)
