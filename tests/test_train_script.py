import contextlib
import copy
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch
from hydra.errors import HydraException

from scripts.train import (
    StageMetadata,
    TrainingJob,
    _validate_checkpoint_metadata,
    main,
)
from t2c_reid.configuration import compose_training_config
from t2c_reid.evaluation import ReIDMetrics
from t2c_reid.wandb import WandbConfig, WandbTracker

RECORDED_CONFIG = None
RECORDED_RNG_STATE = None
RECORDED_EPOCHS: list[int] = []
RECORDED_MODEL_STATE_AT_FIRST_EPOCH: dict = {}
RECORDED_OPTIMIZER_STATE_AT_FIRST_EPOCH: dict = {}


class TrainScriptTest(unittest.TestCase):
    def test_schema2_and_tfc_fingerprint_mismatches_are_rejected(self):
        expected = {
            "schema_version": 3,
            "pid_camera_count_fingerprint": "new-fingerprint",
        }
        with self.assertRaisesRegex(ValueError, "schema_version"):
            _validate_checkpoint_metadata(
                expected,
                {
                    "checkpoint_metadata": {
                        "schema_version": 2,
                        "pid_camera_count_fingerprint": "new-fingerprint",
                    }
                },
            )
        with self.assertRaisesRegex(ValueError, "pid_camera_count_fingerprint"):
            _validate_checkpoint_metadata(
                expected,
                {
                    "checkpoint_metadata": {
                        "schema_version": 3,
                        "pid_camera_count_fingerprint": "old-fingerprint",
                    }
                },
            )

        objective_expected = {
            "schema_version": 3,
            "tfc_weight": 1.0,
            "beta": 0.1,
            "beta_warmup_epochs": 10,
            "stage1_epochs": 20,
            "stage2_first_epoch": 21,
        }
        for key, changed in (
            ("tfc_weight", 0.0),
            ("beta", 0.2),
            ("beta_warmup_epochs", 0),
            ("stage1_epochs", 0),
            ("stage2_first_epoch", 1),
        ):
            actual = dict(objective_expected)
            actual[key] = changed
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, key):
                _validate_checkpoint_metadata(
                    objective_expected,
                    {"checkpoint_metadata": actual},
                )

    def test_seed_override_is_accepted_and_seeds_torch(self):
        global RECORDED_CONFIG, RECORDED_RNG_STATE
        RECORDED_CONFIG = None
        RECORDED_RNG_STATE = None
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_dir = Path(tmp) / "checkpoints"
            main(
                _recording_job_overrides(checkpoint_dir) + ["seed=42"],
                progress_factory=lambda iterable, **kwargs: iterable,
            )

        self.assertEqual(RECORDED_CONFIG.seed, 42)
        # main() must seed torch before calling the job builder, so the RNG
        # state observed inside the builder matches a freshly-seeded generator.
        ref = torch.Generator().manual_seed(42).get_state()
        self.assertTrue(torch.equal(RECORDED_RNG_STATE, ref))

    def test_single_stage_resume_rejects_stage1_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_dir = Path(tmp) / "checkpoints"
            main(
                _job_overrides(checkpoint_dir),
                progress_factory=lambda iterable, **kwargs: iterable,
            )
            payload = torch.load(
                checkpoint_dir / "last.pth",
                map_location="cpu",
                weights_only=True,
            )
            payload["stage"] = "stage1"
            bad_checkpoint = checkpoint_dir / "stage1.pth"
            torch.save(payload, bad_checkpoint)

            with self.assertRaisesRegex(ValueError, "only stage2 checkpoints"):
                main(
                    _job_overrides(
                        checkpoint_dir,
                        epochs=2,
                        resume=bad_checkpoint,
                    ),
                    progress_factory=lambda iterable, **kwargs: iterable,
                )

    def test_resume_restarts_from_saved_checkpoint_epoch(self):
        RECORDED_EPOCHS.clear()
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_dir = Path(tmp) / "checkpoints"
            # Run 2 epochs to produce a last.pth
            main(
                _job_overrides(checkpoint_dir, epochs=2),
                progress_factory=lambda iterable, **kwargs: iterable,
            )

            exit_code = main(
                _job_overrides(
                    checkpoint_dir,
                    builder=f"{__name__}:recording_epoch_builder",
                    epochs=4,
                    resume=checkpoint_dir / "last.pth",
                ),
                progress_factory=lambda iterable, **kwargs: iterable,
            )

        self.assertEqual(exit_code, 0)
        self.assertEqual(RECORDED_EPOCHS, [3, 4])

    def test_resume_restores_model_and_optimizer_state(self):
        RECORDED_EPOCHS.clear()
        RECORDED_MODEL_STATE_AT_FIRST_EPOCH.clear()
        RECORDED_OPTIMIZER_STATE_AT_FIRST_EPOCH.clear()
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_dir = Path(tmp) / "checkpoints"
            main(
                _job_overrides(checkpoint_dir, epochs=2),
                progress_factory=lambda iterable, **kwargs: iterable,
            )
            saved = torch.load(
                checkpoint_dir / "last.pth", map_location="cpu", weights_only=True
            )

            main(
                _job_overrides(
                    checkpoint_dir,
                    builder=f"{__name__}:recording_epoch_builder",
                    epochs=3,
                    resume=checkpoint_dir / "last.pth",
                ),
                progress_factory=lambda iterable, **kwargs: iterable,
            )

        for key, value in saved["model_state"].items():
            self.assertTrue(
                torch.equal(RECORDED_MODEL_STATE_AT_FIRST_EPOCH[key], value)
            )
        saved_momentum = saved["optimizer_state"]["state"][0]["momentum_buffer"]
        recorded_momentum = RECORDED_OPTIMIZER_STATE_AT_FIRST_EPOCH["state"][0][
            "momentum_buffer"
        ]
        self.assertTrue(torch.equal(recorded_momentum, saved_momentum))

    def test_resume_preserves_best_checkpoint_from_previous_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_dir = Path(tmp) / "checkpoints"
            # declining mAP: epoch 1 scores 0.9 and stays the all-time best.
            main(
                _job_overrides(
                    checkpoint_dir,
                    builder=f"{__name__}:declining_map_builder",
                    epochs=2,
                ),
                progress_factory=lambda iterable, **kwargs: iterable,
            )

            main(
                _job_overrides(
                    checkpoint_dir,
                    builder=f"{__name__}:declining_map_builder",
                    epochs=3,
                    resume=checkpoint_dir / "last.pth",
                ),
                progress_factory=lambda iterable, **kwargs: iterable,
            )
            best = torch.load(
                checkpoint_dir / "best.pth", map_location="cpu", weights_only=True
            )

        self.assertEqual(best["epoch"], 1)
        self.assertEqual(best["metrics"]["mAP"], 0.9)

    def test_hydra_defaults_follow_the_project_training_recipe(self):
        config = compose_training_config()

        self.assertEqual(
            config.job_builder,
            "t2c_reid.jobs.siglip2_reid:build_training_job",
        )
        self.assertEqual(config.dataset, "prcc")
        self.assertEqual(config.data_root, Path("data/prcc"))
        self.assertEqual(config.validation_holdout_ids, 20)
        self.assertEqual(config.validation_holdout_seed, 0)
        self.assertEqual(config.stage1_epochs, 60)
        self.assertEqual(config.epochs, 60)
        self.assertEqual(
            config.checkpoint_dir,
            Path("checkpoints/prcc-siglip2-tfc"),
        )
        self.assertEqual(config.run_name, "prcc-siglip2-camera-tfc")

    def test_dataset_config_group_updates_related_paths(self):
        config = compose_training_config(["dataset=market1501"])

        self.assertEqual(config.dataset, "market1501")
        self.assertEqual(config.data_root, Path("data/Market-1501-v15.09.15"))
        self.assertEqual(
            config.checkpoint_dir,
            Path("checkpoints/market1501-siglip2-tfc"),
        )
        self.assertEqual(config.run_name, "market1501-siglip2-camera-tfc")
        self.assertEqual(config.validation_holdout_ids, 0)

    def test_validation_holdout_is_rejected_outside_prcc(self):
        with self.assertRaisesRegex(ValueError, "validation_holdout_ids"):
            compose_training_config(["dataset=msmt17", "validation_holdout_ids=5"])

    def test_prcc_test_split_validation_is_final_epoch_only(self):
        with self.assertRaisesRegex(ValueError, "validation_interval must be >= epochs"):
            compose_training_config(["validation_holdout_ids=0"])

        config = compose_training_config(
            ["validation_holdout_ids=0", "epochs=80", "validation_interval=80"]
        )

        self.assertEqual(config.validation_holdout_ids, 0)

    def test_main_runs_builder_training_job_and_saves_checkpoints(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_dir = Path(tmp) / "checkpoints"
            exit_code = main(
                _job_overrides(checkpoint_dir, epochs=2),
                progress_factory=lambda iterable, **kwargs: iterable,
            )
            best_payload = torch.load(
                checkpoint_dir / "best.pth", map_location="cpu", weights_only=True
            )
            last_payload = torch.load(
                checkpoint_dir / "last.pth", map_location="cpu", weights_only=True
            )

        self.assertEqual(exit_code, 0)
        self.assertEqual(best_payload["epoch"], 2)
        self.assertEqual(last_payload["epoch"], 2)
        self.assertEqual(last_payload["metrics"]["mAP"], 0.2)

    def test_main_passes_project_training_args_to_builder(self):
        global RECORDED_CONFIG
        RECORDED_CONFIG = None
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_dir = Path(tmp) / "checkpoints"
            exit_code = main(
                _recording_job_overrides(checkpoint_dir),
                progress_factory=lambda iterable, **kwargs: iterable,
            )

        self.assertEqual(exit_code, 0)
        self.assertEqual(RECORDED_CONFIG.dataset, "msmt17")
        self.assertEqual(RECORDED_CONFIG.data_root, Path("MSMT17_V1"))
        self.assertEqual(
            RECORDED_CONFIG.siglip2_model_name,
            "google/siglip2-so400m-patch14-384",
        )
        self.assertEqual(RECORDED_CONFIG.batch_size, 8)
        self.assertEqual(RECORDED_CONFIG.eval_batch_size, 128)
        self.assertEqual(RECORDED_CONFIG.gradient_accumulation_steps, 1)
        self.assertEqual(RECORDED_CONFIG.precision, "auto")
        self.assertEqual(RECORDED_CONFIG.num_workers, 2)
        self.assertEqual(RECORDED_CONFIG.data_backend, "rust")
        self.assertEqual(RECORDED_CONFIG.prefetch_factor, 2)
        self.assertIsNone(RECORDED_CONFIG.pin_memory)
        self.assertIsNone(RECORDED_CONFIG.persistent_workers)
        self.assertEqual(RECORDED_CONFIG.rust_data_threads, 2)
        self.assertEqual(RECORDED_CONFIG.evaluation_backend, "rust")
        self.assertEqual(RECORDED_CONFIG.evaluation_chunk_size, 256)
        self.assertEqual(RECORDED_CONFIG.lr, 0.001)
        self.assertEqual(RECORDED_CONFIG.triplet_metric, "euclidean")
        self.assertEqual(RECORDED_CONFIG.device, "cpu")

    def test_default_hyperparameters_follow_siglip2_32gb_recipe(self):
        global RECORDED_CONFIG
        RECORDED_CONFIG = None
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_dir = Path(tmp) / "checkpoints"
            main(
                _job_overrides(
                    checkpoint_dir,
                    builder=f"{__name__}:recording_training_job",
                ),
                progress_factory=lambda iterable, **kwargs: iterable,
            )

        self.assertEqual(RECORDED_CONFIG.image_encoder_lr, 5e-6)
        self.assertEqual(RECORDED_CONFIG.stage1_epochs, 60)
        self.assertEqual(RECORDED_CONFIG.dataset, "prcc")
        self.assertEqual(RECORDED_CONFIG.data_root, Path("data/prcc"))
        # P=16 x K=4: batch-size is the real triplet/SigLIP mining scope and
        # gradient accumulation does not widen it, so accumulation is 1.
        self.assertEqual(RECORDED_CONFIG.batch_size, 64)
        self.assertEqual(RECORDED_CONFIG.num_instances, 4)
        self.assertEqual(RECORDED_CONFIG.gradient_accumulation_steps, 1)
        self.assertEqual(RECORDED_CONFIG.stage1_lr_scheduler, "cosine")
        self.assertEqual(RECORDED_CONFIG.stage1_warmup_epochs, 5)
        self.assertEqual(RECORDED_CONFIG.stage2_lr_scheduler, "cosine")
        self.assertEqual(RECORDED_CONFIG.stage2_warmup_epochs, 5)
        self.assertEqual(RECORDED_CONFIG.reid_head, "bnneck")
        self.assertEqual(RECORDED_CONFIG.label_smoothing, 0.1)
        self.assertEqual(RECORDED_CONFIG.grad_clip_norm, 5.0)
        self.assertFalse(RECORDED_CONFIG.flip_tta)
        # Balanced against the frozen SigLIP calibration t=109.89, b=-15.93.
        self.assertEqual(RECORDED_CONFIG.alignment_weight, 0.1)
        self.assertEqual(RECORDED_CONFIG.tfc_tail_momentum, 0.9)
        self.assertEqual(RECORDED_CONFIG.tfc_class_balance_beta, 0.9999)
        self.assertEqual(RECORDED_CONFIG.tfc_local_weight, 1.0)
        self.assertEqual(RECORDED_CONFIG.tfc_global_weight, 1.0)
        self.assertEqual(RECORDED_CONFIG.tfc_cross_modal_weight, 0.5)
        self.assertEqual(RECORDED_CONFIG.tfc_cross_camera_weight, 0.1)
        self.assertEqual(RECORDED_CONFIG.tfc_contrast_temperature, 0.07)
        self.assertEqual(RECORDED_CONFIG.tfc_transfer_reg_weight, 0.01)
        self.assertTrue(RECORDED_CONFIG.gradient_checkpointing)

    def test_main_logs_validation_metrics_when_wandb_enabled(self):
        run = FakeWandbRun()
        captured: list[tuple[WandbConfig, str]] = []

        @contextlib.contextmanager
        def fake_start_wandb_run(config: WandbConfig, run_name: str):
            captured.append((config, run_name))
            yield WandbTracker(run)

        with tempfile.TemporaryDirectory() as tmp:
            wandb_dir = Path(tmp) / "wandb-data"
            checkpoint_dir = Path(tmp) / "checkpoints"
            with mock.patch("scripts.train.start_wandb_run", new=fake_start_wandb_run):
                exit_code = main(
                    _job_overrides(checkpoint_dir)
                    + [
                        "enable_wandb=true",
                        "wandb_project=T2C-ReID-TrainScript-Test",
                        "wandb_entity=test-team",
                        "wandb_mode=offline",
                        f"wandb_dir={wandb_dir.as_posix()}",
                        "run_name=train-script-test",
                    ],
                    progress_factory=lambda iterable, **kwargs: iterable,
                )

        step_history = [
            payload for payload in run.logged if "stage2_train_step" in payload
        ]
        epoch_history = [payload for payload in run.logged if "stage2_epoch" in payload]
        validation_history = [
            payload for payload in run.logged if "validation_epoch" in payload
        ]
        self.assertEqual(exit_code, 0)
        self.assertEqual(
            captured,
            [
                (
                    WandbConfig(
                        project="T2C-ReID-TrainScript-Test",
                        entity="test-team",
                        mode="offline",
                        directory=wandb_dir,
                    ),
                    "train-script-test",
                )
            ],
        )
        self.assertEqual(
            [payload["stage2_train_step"] for payload in step_history], [1, 2]
        )
        self.assertEqual(
            [payload["stage2_train_step_loss"] for payload in step_history], [1.1, 1.2]
        )
        self.assertEqual(epoch_history[0]["stage2_train_loss"], 1.0)
        self.assertEqual(epoch_history[0]["stage2_lr"], 0.1)
        self.assertEqual(validation_history[0]["mAP"], 0.1)
        self.assertEqual(validation_history[0]["rank_1"], 0.1)
        self.assertEqual(run.config["dataset"], "fixture")

    def test_tracking_disabled_does_not_load_wandb(self):
        with (
            tempfile.TemporaryDirectory() as tmp,
            mock.patch("t2c_reid.wandb._load_wandb") as load_wandb,
        ):
            exit_code = main(
                _job_overrides(Path(tmp) / "checkpoints"),
                progress_factory=lambda iterable, **kwargs: iterable,
            )

        self.assertEqual(exit_code, 0)
        load_wandb.assert_not_called()

    def test_naflex_model_override_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "siglip2_model_name"):
            main(["siglip2_model_name=google/siglip2-so400m-patch16-naflex"])

    def test_removed_clip_model_config_is_rejected(self):
        with self.assertRaises(HydraException):
            main(["clip_model_name=openai/clip-vit-base-patch16"])

    def test_legacy_tracking_config_is_rejected(self):
        legacy_field = "enable_" + "ml" + "flow"
        with self.assertRaises(HydraException):
            main([f"{legacy_field}=true"])


def build_training_job(config) -> TrainingJob:
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

    def train_one_epoch(epoch: int, reporter) -> dict[str, float]:
        for batch_number in reporter.batches([1, 2]):
            optimizer.zero_grad()
            loss = model(torch.eye(2)).sum() * (epoch + batch_number / 10.0)
            loss.backward()
            optimizer.step()
            reporter.report_batch({"loss": epoch + batch_number / 10.0, "lr": 0.1})
        return {"loss": float(epoch), "lr": 0.1}

    def validate(epoch: int) -> ReIDMetrics:
        return ReIDMetrics(map=epoch / 10.0, cmc={1: epoch / 10.0})

    return TrainingJob(
        model,
        optimizer,
        train_one_epoch,
        validate,
        stage_metadata=StageMetadata({"dataset": "fixture", "unused": None}),
    )


def recording_training_job(config) -> TrainingJob:
    global RECORDED_CONFIG, RECORDED_RNG_STATE
    RECORDED_CONFIG = config
    RECORDED_RNG_STATE = torch.get_rng_state()
    return build_training_job(config)


def recording_epoch_builder(config) -> TrainingJob:
    job = build_training_job(config)

    def train_one_epoch(epoch: int, reporter) -> dict[str, float]:
        if not RECORDED_EPOCHS:
            RECORDED_MODEL_STATE_AT_FIRST_EPOCH.update(
                {key: value.clone() for key, value in job.model.state_dict().items()}
            )
            RECORDED_OPTIMIZER_STATE_AT_FIRST_EPOCH.update(
                copy.deepcopy(job.optimizer.state_dict())
            )
        RECORDED_EPOCHS.append(epoch)
        return job.train_one_epoch(epoch, reporter)

    return TrainingJob(job.model, job.optimizer, train_one_epoch, job.validate)


def declining_map_builder(config) -> TrainingJob:
    job = build_training_job(config)

    def validate(epoch: int) -> ReIDMetrics:
        return ReIDMetrics(map=1.0 - epoch / 10.0, cmc={1: 0.5})

    return TrainingJob(job.model, job.optimizer, job.train_one_epoch, validate)


def _recording_job_overrides(checkpoint_dir: Path) -> list[str]:
    return [
        *_job_overrides(
            checkpoint_dir,
            builder=f"{__name__}:recording_training_job",
        ),
        "dataset=msmt17",
        "data_root=MSMT17_V1",
        "siglip2_model_name=google/siglip2-so400m-patch14-384",
        "batch_size=8",
        "num_workers=2",
        "lr=0.001",
        "device=cpu",
    ]


def _job_overrides(
    checkpoint_dir: Path,
    *,
    builder: str | None = None,
    epochs: int = 1,
    validation_interval: int = 1,
    resume: Path | None = None,
) -> list[str]:
    overrides = [
        f"job_builder={builder or f'{__name__}:build_training_job'}",
        f"epochs={epochs}",
        f"validation_interval={validation_interval}",
        f"checkpoint_dir={checkpoint_dir.as_posix()}",
    ]
    if resume is not None:
        overrides.append(f"resume={resume.as_posix()}")
    return overrides


class FakeWandbConfig(dict):
    def update(self, values=None, *, allow_val_change: bool = False):
        super().update({} if values is None else values)


class FakeWandbRun:
    def __init__(self):
        self.config = FakeWandbConfig()
        self.logged: list[dict] = []
        self.defined_metrics: list[tuple[str, dict]] = []

    def define_metric(self, name: str, **kwargs) -> None:
        self.defined_metrics.append((name, dict(kwargs)))

    def log(self, payload) -> None:
        self.logged.append(dict(payload))
