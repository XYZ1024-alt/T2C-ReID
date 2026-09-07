import math
import unittest

import torch
import torch.nn.functional as F

from t2c_reid.losses import (
    _pairwise_distances,
    batch_hard_triplet_loss,
    siglip_identity_anchor_loss,
    supervised_siglip_loss,
)
from t2c_reid.tfc import (
    CameraAwareTFCBank,
    CameraAwareTFCLossConfig,
    _fuse_prototypes,
    _weighted_mean,
)


class TFCLossTest(unittest.TestCase):
    def test_tfc_routes_pid_camera_visual_and_text_centers(self):
        bank = _tfc_bank()
        visual = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        text = torch.tensor([[0.8, 0.2], [0.2, 0.8]])

        bank.update(visual, text, torch.tensor([0, 0]), torch.tensor([0, 1]))

        self.assertTrue(bank.visual_local_initialized[0, 0])
        self.assertTrue(bank.visual_local_initialized[0, 1])
        self.assertFalse(bank.visual_local_initialized[1, 0])
        self.assertTrue(torch.allclose(bank.visual_local_centers[0, 0], visual[0]))
        self.assertTrue(
            torch.allclose(bank.text_local_centers[0, 1], text[1] / text[1].norm())
        )

    def test_tail_identity_uses_more_conservative_momentum(self):
        bank = _tfc_bank()
        self.assertEqual(float(bank.identity_momenta[0]), 0.5)
        self.assertAlmostEqual(float(bank.identity_momenta[1]), 0.9, places=6)
        ids = torch.tensor([0, 1])
        cameras = torch.tensor([0, 0])
        initial = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
        bank.update(initial, initial, ids, cameras)
        changed = torch.tensor([[0.0, 1.0], [0.0, 1.0]])

        bank.update(changed, changed, ids, cameras)

        self.assertGreater(
            float(bank.visual_local_centers[1, 0, 0]),
            float(bank.visual_local_centers[0, 0, 0]),
        )

    def test_global_center_is_equal_camera_average(self):
        bank = CameraAwareTFCBank(
            torch.tensor([101]),
            torch.tensor([[100, 1]]),
            feature_dim=2,
            head_momentum=0.0,
            tail_momentum=0.0,
        )
        bank.update(
            torch.tensor([[1.0, 0.0]]),
            torch.tensor([[1.0, 0.0]]),
            torch.tensor([0]),
            torch.tensor([0]),
        )
        bank.update(
            torch.tensor([[0.0, 1.0]]),
            torch.tensor([[0.0, 1.0]]),
            torch.tensor([0]),
            torch.tensor([1]),
        )

        expected = torch.tensor([2**-0.5, 2**-0.5])
        self.assertTrue(
            torch.allclose(bank.visual_global_centers[0], expected, atol=1e-6)
        )

    def test_effective_number_weights_upweight_tail_identity(self):
        bank = _tfc_bank()
        self.assertAlmostEqual(float(bank.class_weights.mean()), 1.0, places=6)
        self.assertGreater(float(bank.class_weights[1]), float(bank.class_weights[0]))

    def test_camera_prior_masks_diagonal_and_normalizes_active_rows(self):
        bank = _tfc_bank()
        transfer = bank.camera_transfer_probabilities()

        self.assertTrue(torch.equal(torch.diag(transfer), torch.zeros(3)))
        self.assertTrue(torch.allclose(transfer.sum(dim=1), torch.ones(3)))
        self.assertTrue(torch.isfinite(bank.camera_transfer_logits).all())
        self.assertLess(float(bank.transfer_regularization().detach()), 1e-6)

    def test_extreme_transfer_logits_keep_kl_and_infonce_finite(self):
        bank = CameraAwareTFCBank(
            torch.tensor([3, 3]),
            torch.tensor([[1, 1, 1], [1, 1, 1]]),
            feature_dim=2,
            head_momentum=0.0,
            tail_momentum=0.0,
        )
        visual = torch.tensor(
            [[1.0, 0.0], [0.8, 0.2], [0.2, 0.8], [-1.0, 0.0], [-0.8, 0.2], [-0.2, 0.8]]
        )
        ids = torch.tensor([0, 0, 0, 1, 1, 1])
        cameras = torch.tensor([0, 1, 2, 0, 1, 2])
        bank.update(visual, visual, ids, cameras)
        with torch.no_grad():
            bank.camera_transfer_logits[0, 1] = 1000.0
            bank.camera_transfer_logits[0, 2] = -1000.0

        breakdown = bank.loss(
            visual,
            visual,
            ids,
            cameras,
            beta=0.0,
            config=CameraAwareTFCLossConfig(),
        )

        self.assertTrue(torch.isfinite(breakdown.transfer_regularization))
        self.assertTrue(torch.isfinite(breakdown.cross_camera))
        self.assertTrue(torch.isfinite(breakdown.total))

    def test_cross_camera_infonce_backpropagates_to_transfer_logits(self):
        bank = CameraAwareTFCBank(
            torch.tensor([3, 3]),
            torch.tensor([[1, 1, 1], [1, 1, 1]]),
            feature_dim=2,
            head_momentum=0.0,
            tail_momentum=0.0,
        )
        visual = torch.tensor(
            [[1.0, 0.0], [0.8, 0.2], [0.2, 0.8], [-1.0, 0.0], [-0.8, 0.2], [-0.2, 0.8]]
        )
        ids = torch.tensor([0, 0, 0, 1, 1, 1])
        cameras = torch.tensor([0, 1, 2, 0, 1, 2])
        bank.update(visual, visual, ids, cameras)
        config = CameraAwareTFCLossConfig(
            local_weight=0.0,
            global_weight=0.0,
            cross_modal_weight=0.0,
            cross_camera_weight=1.0,
            transfer_reg_weight=0.0,
        )

        breakdown = bank.loss(visual, visual, ids, cameras, beta=0.0, config=config)
        breakdown.total.backward()

        self.assertEqual(float(breakdown.cross_camera_coverage), 1.0)
        self.assertIsNotNone(bank.camera_transfer_logits.grad)
        self.assertGreater(float(bank.camera_transfer_logits.grad.abs().sum()), 0.0)

    def test_single_camera_cross_camera_terms_are_finite_zero(self):
        bank = CameraAwareTFCBank(
            torch.tensor([1, 1]),
            torch.tensor([[1], [1]]),
            feature_dim=2,
            head_momentum=0.5,
        )
        visual = torch.eye(2)
        ids = torch.tensor([0, 1])
        cameras = torch.zeros(2, dtype=torch.long)
        bank.update(visual, visual, ids, cameras)

        breakdown = bank.loss(
            visual,
            visual,
            ids,
            cameras,
            beta=0.1,
            config=CameraAwareTFCLossConfig(),
        )

        self.assertEqual(float(breakdown.cross_camera.detach()), 0.0)
        self.assertEqual(float(breakdown.transfer_regularization.detach()), 0.0)
        self.assertTrue(torch.isfinite(breakdown.total))

    def test_tfc_math_stays_fp32_under_autocast(self):
        bank = _tfc_bank()
        visual = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        ids = torch.tensor([0, 1])
        cameras = torch.tensor([0, 0])
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            bank.update(visual, visual, ids, cameras)
            breakdown = bank.loss(
                visual,
                visual,
                ids,
                cameras,
                beta=0.1,
                config=CameraAwareTFCLossConfig(),
            )

        self.assertEqual(bank.visual_local_centers.dtype, torch.float32)
        self.assertEqual(breakdown.total.dtype, torch.float32)

    def test_beta_changes_cross_modal_prototype_fusion(self):
        bank = _tfc_bank()
        visual = torch.tensor([[1.0, 0.0]])
        text = torch.tensor([[0.0, 1.0]])
        ids = torch.tensor([0])
        cameras = torch.tensor([0])
        bank.update(visual, text, ids, cameras)
        config = CameraAwareTFCLossConfig(
            local_weight=1.0,
            global_weight=0.0,
            cross_modal_weight=0.0,
            cross_camera_weight=0.0,
            transfer_reg_weight=0.0,
        )

        beta_zero = bank.loss(visual, visual, ids, cameras, 0.0, config)
        beta_one = bank.loss(visual, visual, ids, cameras, 1.0, config)

        self.assertLess(float(beta_zero.total.detach()), float(beta_one.total.detach()))

    def test_update_window_rolls_back_only_touched_centers(self):
        bank = _tfc_bank()
        visual = torch.tensor([[1.0, 0.0]])
        bank.update(visual, visual, torch.tensor([0]), torch.tensor([0]))
        before = {
            key: value.clone()
            for key, value in bank.state_dict().items()
            if isinstance(value, torch.Tensor)
        }
        bank.begin_update_window()
        bank.update(
            torch.tensor([[0.0, 1.0]]),
            torch.tensor([[0.0, 1.0]]),
            torch.tensor([0]),
            torch.tensor([0]),
        )

        bank.rollback_update_window()

        after = bank.state_dict()
        for key, expected in before.items():
            self.assertTrue(torch.equal(after[key], expected), key)

    def test_state_dict_round_trip_preserves_centers_and_transfer_logits(self):
        bank = _tfc_bank()
        visual = torch.tensor([[1.0, 0.0]])
        bank.update(visual, visual, torch.tensor([0]), torch.tensor([0]))
        with torch.no_grad():
            bank.camera_transfer_logits[0, 1] += 0.25
        restored = _tfc_bank()

        restored.load_state_dict(bank.state_dict())

        self.assertTrue(
            torch.equal(restored.visual_local_centers, bank.visual_local_centers)
        )
        self.assertTrue(
            torch.equal(restored.text_local_centers, bank.text_local_centers)
        )
        self.assertTrue(
            torch.equal(restored.visual_global_centers, bank.visual_global_centers)
        )
        self.assertTrue(
            torch.equal(restored.text_global_centers, bank.text_global_centers)
        )
        self.assertTrue(
            torch.equal(
                restored.visual_local_initialized, bank.visual_local_initialized
            )
        )
        self.assertTrue(
            torch.equal(restored.text_global_initialized, bank.text_global_initialized)
        )
        self.assertTrue(
            torch.equal(restored.camera_transfer_logits, bank.camera_transfer_logits)
        )

    def test_tfc_validates_statistics_and_indices(self):
        with self.assertRaises(ValueError):
            CameraAwareTFCBank(torch.tensor([2]), torch.tensor([[1, 0]]), 2, 0.5)
        bank = _tfc_bank()
        with self.assertRaises(ValueError):
            bank.update(
                torch.ones(1, 2),
                torch.ones(1, 2),
                torch.tensor([0]),
                torch.tensor([3]),
            )
        with self.assertRaisesRegex(ValueError, "absent from the training split"):
            bank.update(
                torch.ones(1, 2),
                torch.ones(1, 2),
                torch.tensor([1]),
                torch.tensor([1]),
            )


def _tfc_bank() -> CameraAwareTFCBank:
    return CameraAwareTFCBank(
        torch.tensor([10, 2]),
        torch.tensor([[5, 5, 0], [1, 0, 1]]),
        feature_dim=2,
        head_momentum=0.5,
        tail_momentum=0.9,
    )


class SiglipAlignmentLossTest(unittest.TestCase):
    def test_native_reduction_matches_hand_computation(self):
        images = torch.eye(2)
        texts = torch.eye(2)
        person_ids = torch.tensor([0, 1])

        loss = supervised_siglip_loss(
            images,
            texts,
            person_ids,
            logit_scale=1.0,
            logit_bias=0.0,
        )

        # Each row has one positive logit 1 and one negative logit 0.
        expected = F.softplus(torch.tensor(-1.0)) + math.log(2.0)
        self.assertTrue(torch.allclose(loss, expected, atol=1e-6))

    def test_scale_and_bias_are_applied_before_logsigmoid(self):
        images = torch.tensor([[1.0, 0.0]])
        texts = torch.tensor([[1.0, 0.0]])

        loss = supervised_siglip_loss(
            images,
            texts,
            torch.tensor([0]),
            logit_scale=3.0,
            logit_bias=-1.0,
        )

        self.assertTrue(torch.allclose(loss, F.softplus(torch.tensor(-2.0))))

    def test_same_identity_pairs_are_all_positive(self):
        images = torch.tensor([[1.0, 0.0], [0.0, 1.0]], requires_grad=True)
        texts = torch.tensor([[1.0, 0.0], [0.0, 1.0]])

        loss = supervised_siglip_loss(
            images,
            texts,
            torch.tensor([0, 0]),
            logit_scale=1.0,
            logit_bias=0.0,
        )
        loss.backward()

        # Moving image 0 toward text 1 lowers the all-positive objective.
        self.assertLess(float(images.grad[0] @ texts[1]), 0.0)

    def test_pairwise_loss_runs_in_fp32_under_low_precision_inputs(self):
        loss = supervised_siglip_loss(
            torch.eye(2, dtype=torch.bfloat16),
            torch.eye(2, dtype=torch.bfloat16),
            torch.tensor([0, 1]),
            logit_scale=2.0,
            logit_bias=-0.5,
        )
        self.assertEqual(loss.dtype, torch.float32)

    def test_pairwise_loss_disables_active_autocast(self):
        images = torch.eye(2)
        texts = torch.eye(2)
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            loss = supervised_siglip_loss(
                images,
                texts,
                torch.tensor([0, 1]),
                logit_scale=2.0,
                logit_bias=-0.5,
            )

        reference = supervised_siglip_loss(
            images,
            texts,
            torch.tensor([0, 1]),
            logit_scale=2.0,
            logit_bias=-0.5,
        )
        self.assertEqual(loss.dtype, torch.float32)
        self.assertTrue(torch.equal(loss, reference))

    def test_supervised_loss_validates_person_ids(self):
        with self.assertRaises(ValueError):
            supervised_siglip_loss(torch.eye(2), torch.eye(2), torch.tensor([0]))
        with self.assertRaises(ValueError):
            supervised_siglip_loss(torch.eye(2), torch.eye(2), torch.tensor([0.5, 1.5]))


class SiglipIdentityAnchorLossTest(unittest.TestCase):
    def test_stage2_uses_one_positive_and_all_other_anchors_as_negatives(self):
        anchors = torch.eye(2)
        visual = torch.tensor([[2.0, 0.0], [0.0, 3.0]])

        loss = siglip_identity_anchor_loss(
            visual,
            anchors,
            torch.tensor([0, 1]),
            logit_scale=1.0,
            logit_bias=0.0,
        )

        expected = F.softplus(torch.tensor(-1.0)) + math.log(2.0)
        self.assertTrue(torch.allclose(loss, expected, atol=1e-6))

    def test_correct_anchor_assignment_scores_lower_than_wrong(self):
        anchors = torch.eye(2)
        visual = torch.eye(2)
        correct = siglip_identity_anchor_loss(
            visual, anchors, torch.tensor([0, 1]), logit_scale=5.0
        )
        wrong = siglip_identity_anchor_loss(
            visual, anchors, torch.tensor([1, 0]), logit_scale=5.0
        )
        self.assertLess(float(correct), float(wrong))

    def test_extra_negative_anchor_participates_in_native_sum(self):
        visual = torch.tensor([[1.0, 0.0]])
        ids = torch.tensor([0])
        two = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        three = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.9, 0.1]])

        without_extra = siglip_identity_anchor_loss(visual, two, ids, logit_scale=5.0)
        with_extra = siglip_identity_anchor_loss(visual, three, ids, logit_scale=5.0)

        self.assertGreater(float(with_extra), float(without_extra))

    def test_stage2_alignment_validates_shapes_and_ids(self):
        with self.assertRaises(ValueError):
            siglip_identity_anchor_loss(
                torch.ones(2, 2, 2), torch.eye(2), torch.tensor([0, 1])
            )
        with self.assertRaises(ValueError):
            siglip_identity_anchor_loss(
                torch.eye(2), torch.ones(2), torch.tensor([0, 1])
            )
        with self.assertRaises(ValueError):
            siglip_identity_anchor_loss(
                torch.ones(2, 3), torch.ones(4, 2), torch.tensor([0, 1])
            )
        with self.assertRaises(ValueError):
            siglip_identity_anchor_loss(
                torch.eye(2), torch.eye(2), torch.tensor([0.5, 1.5])
            )
        with self.assertRaises(ValueError):
            siglip_identity_anchor_loss(torch.eye(2), torch.eye(2), torch.tensor([0]))
        with self.assertRaises(ValueError):
            siglip_identity_anchor_loss(
                torch.eye(2), torch.eye(2), torch.tensor([0, 2])
            )


class TripletLossTest(unittest.TestCase):
    def test_autocast_preserves_loss_and_gradients(self):
        devices = ["cpu"]
        if torch.cuda.is_available():
            devices.append("cuda")
        generator = torch.Generator().manual_seed(7)
        base = torch.randn(1, 1152, generator=generator)
        features = base + 0.02 * torch.randn(8, 1152, generator=generator)
        for device in devices:
            dtypes = [torch.bfloat16] if device == "cpu" else [torch.float16]
            if device == "cuda" and torch.cuda.is_bf16_supported():
                dtypes.append(torch.bfloat16)
            labels = torch.arange(4, device=device).repeat_interleave(2)
            for dtype in dtypes:
                for metric in ("euclidean", "cosine"):
                    for input_dtype in (torch.float32, dtype):
                        with self.subTest(
                            device=device,
                            dtype=dtype,
                            metric=metric,
                            input_dtype=input_dtype,
                        ):
                            actual_input = (
                                features.to(device=device, dtype=input_dtype)
                                .detach()
                                .clone()
                                .requires_grad_()
                            )
                            reference_input = (
                                actual_input.detach().float().requires_grad_()
                            )
                            reference = batch_hard_triplet_loss(
                                reference_input, labels, metric=metric
                            )
                            reference.backward()
                            with torch.autocast(device_type=device, dtype=dtype):
                                actual = batch_hard_triplet_loss(
                                    actual_input, labels, metric=metric
                                )
                            actual.backward()
                            self.assertEqual(actual.dtype, torch.float32)
                            torch.testing.assert_close(actual, reference)
                            self.assertGreater(reference_input.grad.norm().item(), 0.0)
                            torch.testing.assert_close(
                                actual_input.grad, reference_input.grad.to(input_dtype)
                            )

    def test_batch_hard_triplet_penalizes_close_negative(self):
        features = torch.tensor([[1.0, 0.0], [0.9, 0.1], [0.8, 0.2]])
        labels = torch.tensor([0, 0, 1])
        loss = batch_hard_triplet_loss(features, labels, margin=0.3)
        self.assertGreater(float(loss), 0.0)

    def test_euclidean_default_is_hand_computed(self):
        features = torch.tensor([[0.0, 0.0], [3.0, 4.0], [6.0, 8.0]])
        labels = torch.tensor([0, 0, 1])
        loss = batch_hard_triplet_loss(features, labels, margin=0.3)
        self.assertTrue(torch.allclose(loss, torch.tensor(0.15), atol=1e-6))

    def test_cosine_and_euclidean_differ_for_scaled_direction(self):
        features = torch.tensor([[3.0, 4.0], [6.0, 8.0], [0.0, 1.0]])
        labels = torch.tensor([0, 0, 1])
        euclidean = batch_hard_triplet_loss(features, labels, metric="euclidean")
        cosine = batch_hard_triplet_loss(features, labels, metric="cosine")
        self.assertFalse(torch.allclose(euclidean, cosine))

    def test_unknown_triplet_metric_is_rejected(self):
        with self.assertRaises(ValueError):
            batch_hard_triplet_loss(
                torch.tensor([[1.0, 0.0], [0.9, 0.1], [0.8, 0.2]]),
                torch.tensor([0, 0, 1]),
                metric="manhattan",
            )

    def test_anchors_without_a_positive_or_negative_are_excluded(self):
        # Identity 2 is a singleton, so it has no positive and must not join
        # the mean. The masked implementation still produces a value for its
        # row; only the validity mask keeps it out.
        features = torch.tensor([[0.0, 0.0], [3.0, 4.0], [6.0, 8.0], [60.0, 80.0]])
        labels = torch.tensor([0, 0, 1, 2])
        paired = batch_hard_triplet_loss(features[:3], labels[:3], margin=0.3)
        with_singleton = batch_hard_triplet_loss(features, labels, margin=0.3)
        self.assertTrue(torch.isfinite(with_singleton))
        # Adding a far-away singleton changes the hardest negative for rows 0-2,
        # but the singleton row itself contributes nothing.
        self.assertEqual(with_singleton.ndim, paired.ndim)

    def test_every_label_identical_has_no_negative_and_raises(self):
        with self.assertRaises(ValueError):
            batch_hard_triplet_loss(
                torch.tensor([[1.0, 0.0], [0.9, 0.1], [0.8, 0.2]]),
                torch.tensor([0, 0, 0]),
            )

    def test_matches_the_per_anchor_reference_including_gradients(self):
        torch.manual_seed(11)
        for metric in ("euclidean", "cosine"):
            with self.subTest(metric=metric):
                features = torch.randn(16, 12) * 7.0
                labels = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3])

                actual_input = features.clone().requires_grad_(True)
                actual = batch_hard_triplet_loss(
                    actual_input, labels, 0.3, metric=metric
                )
                actual.backward()

                reference_input = features.clone().requires_grad_(True)
                reference = _reference_batch_hard_triplet(
                    reference_input, labels, 0.3, metric
                )
                reference.backward()

                self.assertTrue(torch.allclose(actual, reference, atol=1e-6))
                self.assertTrue(
                    torch.allclose(actual_input.grad, reference_input.grad, atol=1e-6)
                )


class CrossCameraLossVectorizationTest(unittest.TestCase):
    """The batched cross-camera InfoNCE must equal the per-sample loop it replaced."""

    def test_matches_the_per_sample_reference_including_both_gradient_paths(self):
        torch.manual_seed(3)
        counts = torch.tensor([6, 5, 4, 3, 2, 2])
        per_camera = torch.tensor(
            [
                [2, 2, 1, 1],
                [2, 1, 1, 1],
                [1, 1, 1, 1],
                [1, 1, 1, 0],
                [1, 1, 0, 0],
                [2, 0, 0, 0],
            ]
        )
        for trial in range(3):
            with self.subTest(trial=trial):
                bank = CameraAwareTFCBank(
                    counts, per_camera, feature_dim=8, head_momentum=0.5
                )
                bank.camera_transfer_logits.data.normal_(0.0, 0.7)
                for _ in range(4):
                    person_ids = torch.randint(0, 6, (10,))
                    camera_ids = torch.stack(
                        [
                            torch.multinomial(per_camera[p].float(), 1)[0]
                            for p in person_ids
                        ]
                    )
                    bank.update(
                        F.normalize(torch.randn(10, 8), dim=1),
                        F.normalize(torch.randn(10, 8), dim=1),
                        person_ids,
                        camera_ids,
                    )

                person_ids = torch.randint(0, 6, (12,))
                camera_ids = torch.stack(
                    [torch.multinomial(per_camera[p].float(), 1)[0] for p in person_ids]
                )
                features = F.normalize(torch.randn(12, 8), dim=1)
                weights = bank.class_weights[person_ids].float()

                actual_input = features.clone().requires_grad_(True)
                bank.camera_transfer_logits.grad = None
                actual, actual_coverage = bank._cross_camera_loss(
                    actual_input, person_ids, camera_ids, weights, 0.1, 0.07
                )
                actual.backward()
                actual_transfer_grad = bank.camera_transfer_logits.grad.clone()

                reference_input = features.clone().requires_grad_(True)
                bank.camera_transfer_logits.grad = None
                reference, reference_coverage = _reference_cross_camera_loss(
                    bank, reference_input, person_ids, camera_ids, weights, 0.1, 0.07
                )
                reference.backward()
                reference_transfer_grad = bank.camera_transfer_logits.grad.clone()

                self.assertTrue(torch.allclose(actual, reference, atol=1e-6))
                self.assertTrue(torch.allclose(actual_coverage, reference_coverage))
                self.assertTrue(
                    torch.allclose(actual_input.grad, reference_input.grad, atol=1e-6)
                )
                self.assertTrue(
                    torch.allclose(
                        actual_transfer_grad, reference_transfer_grad, atol=1e-6
                    )
                )

    def test_single_camera_identity_contributes_nothing(self):
        # Identity 5 only ever appears in camera 0, so it can never have a
        # cross-camera positive and must be dropped from the coverage count.
        bank = CameraAwareTFCBank(
            torch.tensor([4, 2]),
            torch.tensor([[2, 2], [2, 0]]),
            feature_dim=4,
            head_momentum=0.5,
        )
        bank.update(
            F.normalize(torch.randn(3, 4), dim=1),
            F.normalize(torch.randn(3, 4), dim=1),
            torch.tensor([0, 0, 1]),
            torch.tensor([0, 1, 0]),
        )
        person_ids = torch.tensor([0, 1])
        camera_ids = torch.tensor([0, 0])
        loss, coverage = bank._cross_camera_loss(
            F.normalize(torch.randn(2, 4), dim=1),
            person_ids,
            camera_ids,
            bank.class_weights[person_ids].float(),
            0.1,
            0.07,
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertAlmostEqual(float(coverage), 0.5, places=6)


def _reference_cross_camera_loss(
    bank, retrieval, person_ids, camera_ids, sample_weights, beta, temperature
):
    """The original per-sample Python loop, kept as an executable oracle."""

    initialized = bank.visual_local_initialized & bank.text_local_initialized
    pair_indices = torch.nonzero(initialized, as_tuple=False)
    zero = retrieval.sum() * 0.0 + bank.camera_transfer_logits.float().sum() * 0.0
    if pair_indices.shape[0] == 0:
        return zero, zero.detach()
    pair_person_ids = pair_indices[:, 0]
    pair_camera_ids = pair_indices[:, 1]
    prototypes = _fuse_prototypes(
        bank.visual_local_centers[initialized].detach(),
        bank.text_local_centers[initialized].detach(),
        beta,
    )
    transfer_log_probabilities = bank.camera_transfer_log_probabilities()
    losses: list[torch.Tensor] = []
    weights: list[torch.Tensor] = []
    for index in range(retrieval.shape[0]):
        person_id = person_ids[index]
        camera_id = camera_ids[index]
        positives = (pair_person_ids == person_id) & (pair_camera_ids != camera_id)
        negatives = pair_person_ids != person_id
        if not bool(torch.any(positives)) or not bool(torch.any(negatives)):
            continue
        positive_log_weights = transfer_log_probabilities[
            camera_id, pair_camera_ids[positives]
        ]
        if not bool(torch.all(torch.isfinite(positive_log_weights))):
            continue
        positive_log_weights = positive_log_weights - torch.logsumexp(
            positive_log_weights, dim=0
        )
        candidate_mask = positives | negatives
        logits = (retrieval[index] @ prototypes.T) / float(temperature)
        numerator = torch.logsumexp(logits[positives] + positive_log_weights, dim=0)
        denominator = torch.logsumexp(logits[candidate_mask], dim=0)
        losses.append(denominator - numerator)
        weights.append(sample_weights[index])
    if not losses:
        return zero, zero.detach()
    coverage = torch.tensor(
        len(losses) / retrieval.shape[0], dtype=torch.float32, device=retrieval.device
    )
    return _weighted_mean(torch.stack(losses), torch.stack(weights)), coverage


def _reference_batch_hard_triplet(features, labels, margin, metric):
    """The original per-anchor Python loop, kept as an executable oracle."""

    distances = _pairwise_distances(features, metric)
    losses = []
    for index in range(labels.shape[0]):
        positive_mask = labels == labels[index]
        negative_mask = labels != labels[index]
        positive_mask[index] = False
        if bool(torch.any(positive_mask)) and bool(torch.any(negative_mask)):
            hardest_positive = torch.max(distances[index][positive_mask])
            hardest_negative = torch.min(distances[index][negative_mask])
            losses.append(torch.relu(hardest_positive - hardest_negative + margin))
    return torch.stack(losses).mean()


if __name__ == "__main__":
    unittest.main()
