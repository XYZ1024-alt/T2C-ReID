import unittest

import torch

from t2c_reid.features import l2_normalize
from t2c_reid.losses import (
    batch_hard_triplet,
    batch_hard_triplet_loss,
    siglip_identity_anchor_loss,
    supervised_siglip_loss,
)
from t2c_reid.model import T2CReIDModel
from t2c_reid.prompts import PromptBank, PromptConfig
from t2c_reid.tfc import CameraAwareTFCBank
from t2c_reid.training import (
    Stage1LossConfig,
    Stage2LossConfig,
    Stage2LossInputs,
    TrainingBatch,
    stage1_alignment_loss,
    stage1_alignment_loss_from_visual,
    stage2_loss_breakdown,
)


class TFCCenterBank(CameraAwareTFCBank):
    """Test adapter for legacy fixtures; production exposes CameraAwareTFCBank only."""

    def __init__(self, num_train_ids: int, feature_dim: int, momentum: float):
        super().__init__(
            identity_counts=torch.full((num_train_ids,), 2, dtype=torch.long),
            identity_camera_counts=torch.ones(num_train_ids, 2, dtype=torch.long),
            feature_dim=feature_dim,
            head_momentum=momentum,
            tail_momentum=momentum,
        )

    def update(self, features, labels, person_ids=None, camera_ids=None):
        if person_ids is None or camera_ids is None:
            return
        return super().update(features, labels, person_ids, camera_ids)

    @property
    def centers(self):
        return self.visual_global_centers

    @property
    def initialized(self):
        return self.visual_global_initialized


class IdentityEncoder(torch.nn.Module):
    def forward(self, inputs, camera_ids=None):
        return inputs


class PromptMeanEncoder(torch.nn.Module):
    def forward(self, prompts):
        return prompts.mean(dim=1)


class TrainingLossTest(unittest.TestCase):
    def test_stage1_alignment_uses_training_identity_prompts(self):
        model = _training_model(beta=0.0)
        batch = TrainingBatch(
            images=torch.eye(2),
            camera_ids=torch.tensor([0, 1]),
            person_ids=torch.tensor([0, 1]),
        )

        breakdown = stage1_alignment_loss(model, batch, Stage1LossConfig(logit_scale=10.0))

        # Native SigLIP sums the negative pair's softplus(0), so the matched
        # two-sample fixture approaches ln(2), not zero.
        self.assertLess(float(breakdown.total.detach()), 0.7)
        self.assertTrue(torch.allclose(breakdown.total, breakdown.alignment))

    def test_stage1_alignment_treats_same_identity_as_positive(self):
        # PK-sampled batches repeat identities; the stage-1 alignment loss must
        # score same-person rows as positives, not arange negatives.
        model = _training_model(beta=0.0)
        batch = TrainingBatch(
            images=torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.7, 0.7]]),
            camera_ids=torch.tensor([0, 1, 0]),
            person_ids=torch.tensor([0, 0, 1]),
        )
        config = Stage1LossConfig(logit_scale=5.0)
        outputs = model.forward_stage1(batch.images, batch.camera_ids, batch.person_ids)
        expected = supervised_siglip_loss(
            outputs["visual"], outputs["text"], batch.person_ids, logit_scale=5.0
        )

        breakdown = stage1_alignment_loss(model, batch, config)

        self.assertTrue(torch.allclose(breakdown.alignment, expected))

    def test_stage1_alignment_loss_from_visual_matches_full_loss(self):
        # The cached Stage-1 path feeds a precomputed normalized image feature
        # into the shared loss helper; it must match the online loss bitwise.
        model = _training_model(beta=0.0)
        batch = TrainingBatch(
            images=torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.7, 0.7]]),
            camera_ids=torch.tensor([0, 1, 0]),
            person_ids=torch.tensor([0, 0, 1]),
        )
        config = Stage1LossConfig(logit_scale=5.0)
        visual = model.encode_visual(batch.images, batch.camera_ids)
        expected = stage1_alignment_loss(model, batch, config)

        breakdown = stage1_alignment_loss_from_visual(
            model,
            visual,
            camera_ids=batch.camera_ids,
            person_ids=batch.person_ids,
            config=config,
        )

        self.assertTrue(torch.equal(breakdown.alignment, expected.alignment))

    def test_stage2_alignment_classifies_visual_against_anchor_matrix(self):
        # SigLIP 2 Stage-2: the L2-normalized image feature is classified
        # against the frozen identity anchor matrix over ALL train ids.
        model = _training_model(beta=0.0)
        classifier = torch.nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            classifier.weight.copy_(torch.eye(2))
        batch = TrainingBatch(
            images=torch.tensor([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]]),
            camera_ids=torch.tensor([0, 1, 0]),
            person_ids=torch.tensor([0, 0, 1]),
        )
        anchors = torch.tensor([[0.8, 0.6], [0.6, -0.8]])
        tfc_bank = TFCCenterBank(num_train_ids=2, feature_dim=2, momentum=0.5)
        tfc_bank.update(model.forward_stage2(batch.images, batch.camera_ids, batch.person_ids)["retrieval"], batch.person_ids)
        outputs = model.forward_stage2(batch.images, batch.camera_ids, batch.person_ids)
        expected = siglip_identity_anchor_loss(
            outputs["visual"], anchors, batch.person_ids, logit_scale=Stage2LossConfig().logit_scale
        )

        breakdown = stage2_loss_breakdown(
            model,
            batch,
            Stage2LossInputs(classifier=classifier, tfc_bank=tfc_bank, anchors=anchors),
        )

        self.assertTrue(torch.allclose(breakdown.alignment, expected))

    def test_stage2_alignment_ignores_reid_label_smoothing(self):
        model = _training_model(beta=0.0)
        classifier = torch.nn.Linear(2, 2, bias=False)
        batch = TrainingBatch(
            images=torch.tensor([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]]),
            camera_ids=torch.tensor([0, 0, 1, 1]),
            person_ids=torch.tensor([0, 0, 1, 1]),
        )
        anchors = torch.eye(2)
        breakdowns = {}
        for smoothing in (0.0, 0.4):
            tfc_bank = TFCCenterBank(num_train_ids=2, feature_dim=2, momentum=0.5)
            breakdowns[smoothing] = stage2_loss_breakdown(
                model,
                batch,
                Stage2LossInputs(
                    classifier=classifier,
                    tfc_bank=tfc_bank,
                    anchors=anchors,
                    config=Stage2LossConfig(logit_scale=20.0, label_smoothing=smoothing),
                ),
            )

        self.assertTrue(
            torch.equal(
                breakdowns[0.4].alignment,
                breakdowns[0.0].alignment,
            )
        )
        self.assertNotEqual(
            float(breakdowns[0.4].identity.detach()),
            float(breakdowns[0.0].identity.detach()),
        )

    def test_stage2_loss_breakdown_combines_alignment_reid_and_tfc(self):
        model = _training_model(beta=0.0)
        classifier = torch.nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            classifier.weight.copy_(10.0 * torch.eye(2))
        batch = TrainingBatch(
            images=torch.tensor([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]]),
            camera_ids=torch.tensor([0, 0, 1]),
            person_ids=torch.tensor([0, 0, 1]),
        )
        tfc_bank = TFCCenterBank(num_train_ids=2, feature_dim=2, momentum=0.5)
        tfc_bank.update(model.forward_stage2(batch.images, batch.camera_ids, batch.person_ids)["retrieval"], batch.person_ids)
        inputs = Stage2LossInputs(
            classifier=classifier,
            tfc_bank=tfc_bank,
            anchors=torch.eye(2),
            config=Stage2LossConfig(tfc_weight=0.5, alignment_weight=1.0),
        )

        breakdown = stage2_loss_breakdown(model, batch, inputs)

        expected = (
            breakdown.identity + breakdown.triplet
            + breakdown.alignment_weight * breakdown.alignment
            + breakdown.tfc_weight * breakdown.tfc
        )
        self.assertTrue(torch.allclose(breakdown.total, expected))
        self.assertEqual(breakdown.alignment_weight, 1.0)
        self.assertEqual(breakdown.tfc_weight, 0.5)

    def test_stage2_loss_uses_configured_label_smoothing(self):
        model = _training_model(beta=0.0)
        classifier = torch.nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            classifier.weight.copy_(torch.eye(2))
        batch = TrainingBatch(
            images=torch.tensor([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]]),
            camera_ids=torch.tensor([0, 0, 1, 1]),
            person_ids=torch.tensor([0, 0, 1, 1]),
        )
        tfc_bank = TFCCenterBank(num_train_ids=2, feature_dim=2, momentum=0.5)
        tfc_bank.update(model.forward_stage2(batch.images, batch.camera_ids, batch.person_ids)["retrieval"], batch.person_ids)
        config = Stage2LossConfig(label_smoothing=0.1)

        breakdown = stage2_loss_breakdown(
            model,
            batch,
            Stage2LossInputs(classifier=classifier, tfc_bank=tfc_bank, anchors=torch.eye(2), config=config),
        )

        self.assertGreater(float(breakdown.identity.detach()), 0.0)

    def test_stage2_loss_scales_identity_logits(self):
        model = _training_model(beta=0.0)
        classifier = torch.nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            classifier.weight.copy_(torch.eye(2))
        batch = TrainingBatch(
            images=torch.tensor([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]]),
            camera_ids=torch.tensor([0, 0, 1, 1]),
            person_ids=torch.tensor([0, 0, 1, 1]),
        )
        tfc_bank = TFCCenterBank(num_train_ids=2, feature_dim=2, momentum=0.5)
        tfc_bank.update(model.forward_stage2(batch.images, batch.camera_ids, batch.person_ids)["retrieval"], batch.person_ids)

        unscaled = stage2_loss_breakdown(
            model,
            batch,
            Stage2LossInputs(classifier=classifier, tfc_bank=tfc_bank, anchors=torch.eye(2)),
        )
        scaled = stage2_loss_breakdown(
            model,
            batch,
            Stage2LossInputs(
                classifier=classifier,
                tfc_bank=tfc_bank,
                anchors=torch.eye(2),
                config=Stage2LossConfig(id_logit_scale=10.0),
            ),
        )

        self.assertLess(float(scaled.identity.detach()), float(unscaled.identity.detach()))

    def test_stage2_loss_classifies_feature_head_output(self):
        head = torch.nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            head.weight.copy_(torch.tensor([[0.0, 1.0], [1.0, 0.0]]))
        model = _training_model(beta=0.0, feature_head=head)
        classifier = torch.nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            classifier.weight.copy_(10.0 * torch.eye(2))
        batch = TrainingBatch(
            images=torch.tensor([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]]),
            camera_ids=torch.tensor([0, 0, 1, 1]),
            person_ids=torch.tensor([1, 1, 0, 0]),
        )
        tfc_bank = TFCCenterBank(num_train_ids=2, feature_dim=2, momentum=0.5)
        tfc_bank.update(model.forward_stage2(batch.images, batch.camera_ids, batch.person_ids)["retrieval"], batch.person_ids)

        breakdown = stage2_loss_breakdown(
            model,
            batch,
            Stage2LossInputs(classifier=classifier, tfc_bank=tfc_bank, anchors=torch.eye(2)),
        )

        self.assertLess(float(breakdown.identity.detach()), 0.01)

    def test_stage2_identity_logits_are_independent_of_fused_text(self):
        # The ID CE must act on the BNNeck output of the image feature, never
        # on the fused retrieval feature: changing beta (and thus the fused
        # text contribution) must leave the identity loss bitwise unchanged.
        classifier = torch.nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            classifier.weight.copy_(torch.eye(2))
        batch = TrainingBatch(
            images=torch.tensor([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]]),
            camera_ids=torch.tensor([0, 1, 0, 1]),
            person_ids=torch.tensor([0, 0, 1, 1]),
        )
        losses = []
        for beta in (0.0, 5.0):
            model = _training_model(beta=beta, camera_prompt=torch.tensor([[0.6, 0.8]]))
            tfc_bank = TFCCenterBank(num_train_ids=2, feature_dim=2, momentum=0.5)
            breakdown = stage2_loss_breakdown(
                model,
                batch,
                Stage2LossInputs(classifier=classifier, tfc_bank=tfc_bank, anchors=torch.eye(2)),
            )
            losses.append(breakdown.identity)
        # Sanity: a nonzero camera prompt with beta>0 does change the fused feature.
        beta0 = _training_model(beta=0.0, camera_prompt=torch.tensor([[0.6, 0.8]]))
        beta5 = _training_model(beta=5.0, camera_prompt=torch.tensor([[0.6, 0.8]]))
        self.assertFalse(torch.allclose(
            beta0.forward_stage2(batch.images, batch.camera_ids, batch.person_ids)["retrieval"],
            beta5.forward_stage2(batch.images, batch.camera_ids, batch.person_ids)["retrieval"],
        ))

        self.assertTrue(torch.equal(losses[0], losses[1]))

    def test_stage2_triplet_uses_pre_bnneck_feature(self):
        # Triplet must consume the BN-PRE image feature (visual_raw), not the
        # feature-head output. The head zeroes dimension 1, which collapses
        # same-id pairs onto negatives and would drive the loss to zero.
        head = torch.nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            head.weight.copy_(torch.tensor([[1.0, 0.0], [0.0, 0.0]]))
        model = _training_model(beta=0.0, feature_head=head)
        classifier = torch.nn.Linear(2, 2, bias=False)
        batch = TrainingBatch(
            images=torch.tensor([[1.0, 2.0], [1.0, -2.0], [3.0, 2.0], [3.0, -2.0]]),
            camera_ids=torch.tensor([0, 0, 1, 1]),
            person_ids=torch.tensor([0, 0, 1, 1]),
        )
        tfc_bank = TFCCenterBank(num_train_ids=2, feature_dim=2, momentum=0.5)
        config = Stage2LossConfig(triplet_metric="euclidean")

        breakdown = stage2_loss_breakdown(
            model,
            batch,
            Stage2LossInputs(classifier=classifier, tfc_bank=tfc_bank, anchors=torch.eye(2), config=config),
        )

        expected = batch_hard_triplet_loss(
            batch.images, batch.person_ids, margin=config.triplet_margin, metric="euclidean"
        )
        headed = batch_hard_triplet_loss(
            head(batch.images), batch.person_ids, margin=config.triplet_margin, metric="euclidean"
        )
        self.assertTrue(torch.allclose(breakdown.triplet, expected))
        self.assertGreater(float(expected.detach()), 0.0)
        self.assertFalse(torch.allclose(breakdown.triplet, headed))
        self.assertGreater(float(breakdown.triplet_active_fraction), 0.0)
        self.assertFalse(breakdown.triplet_active_fraction.requires_grad)

    def test_triplet_active_fraction_counts_anchors_with_positive_hinge(self):
        features = torch.tensor([[0.0], [0.1], [10.0], [10.1]], requires_grad=True)
        labels = torch.tensor([0, 0, 1, 1])

        inert = batch_hard_triplet(features, labels, margin=0.3)
        active = batch_hard_triplet(features, labels, margin=20.0)

        self.assertEqual(float(inert.active_fraction), 0.0)
        self.assertEqual(float(active.active_fraction), 1.0)
        self.assertTrue(torch.equal(inert.loss, batch_hard_triplet_loss(features, labels, 0.3)))

    def test_stage2_triplet_uses_configured_metric(self):
        model = _training_model(beta=0.0)
        classifier = torch.nn.Linear(2, 2, bias=False)
        batch = TrainingBatch(
            images=torch.tensor([[0.0, 0.0], [3.0, 4.0], [6.0, 8.0], [0.0, 1.0]]),
            camera_ids=torch.tensor([0, 0, 1, 1]),
            person_ids=torch.tensor([0, 0, 1, 1]),
        )

        breakdowns = {}
        for metric in ("euclidean", "cosine"):
            tfc_bank = TFCCenterBank(num_train_ids=2, feature_dim=2, momentum=0.5)
            breakdowns[metric] = stage2_loss_breakdown(
                model,
                batch,
                Stage2LossInputs(
                    classifier=classifier,
                    tfc_bank=tfc_bank,
                    anchors=torch.eye(2),
                    config=Stage2LossConfig(triplet_metric=metric),
                ),
            )

        for metric in ("euclidean", "cosine"):
            expected = batch_hard_triplet_loss(
                batch.images, batch.person_ids, margin=Stage2LossConfig().triplet_margin, metric=metric
            )
            self.assertTrue(torch.allclose(breakdowns[metric].triplet, expected))
        self.assertFalse(torch.allclose(breakdowns["euclidean"].triplet, breakdowns["cosine"].triplet))

    def test_stage2_loss_updates_tfc_centers_from_retrieval_feature(self):
        # Stage-2 must initialize/update TFC centers from the SAME forward's
        # fused retrieval feature before scoring — no separate no-grad forward
        # and no RuntimeError on a fresh center bank. With beta=0 the retrieval
        # feature is l2n(head(visual_raw)); the head zeroes dimension 1.
        head = torch.nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            head.weight.copy_(torch.tensor([[1.0, 0.0], [0.0, 0.0]]))
        model = _training_model(beta=0.0, feature_head=head)
        classifier = torch.nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            classifier.weight.copy_(torch.eye(2))
        batch = TrainingBatch(
            images=torch.tensor([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]]),
            camera_ids=torch.tensor([0, 0, 1]),
            person_ids=torch.tensor([0, 0, 1]),
        )
        tfc_bank = TFCCenterBank(num_train_ids=2, feature_dim=2, momentum=0.5)

        breakdown = stage2_loss_breakdown(
            model,
            batch,
            Stage2LossInputs(classifier=classifier, tfc_bank=tfc_bank, anchors=torch.eye(2)),
        )

        self.assertTrue(bool(tfc_bank.initialized.all()))
        # Centers live in retrieval space: the head zeroes dimension 1.
        self.assertTrue(torch.allclose(tfc_bank.centers[:, 1], torch.zeros(2)))
        self.assertGreaterEqual(float(breakdown.tfc.detach()), 0.0)

    def test_stage2_tfc_uses_retrieval_feature(self):
        head = torch.nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            head.weight.copy_(torch.tensor([[1.0, 0.0], [0.0, 0.0]]))
        model = _training_model(beta=0.0, feature_head=head)
        classifier = torch.nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            classifier.weight.copy_(torch.eye(2))
        batch = TrainingBatch(
            images=torch.tensor([[1.0, 0.5], [1.0, -0.5], [-1.0, 0.5], [-1.0, -0.5]]),
            camera_ids=torch.tensor([0, 0, 1, 1]),
            person_ids=torch.tensor([0, 0, 1, 1]),
        )
        retrieval = model.forward_stage2(batch.images, batch.camera_ids, batch.person_ids)["retrieval"]
        tfc_bank = TFCCenterBank(num_train_ids=2, feature_dim=2, momentum=0.5)
        tfc_bank.update(retrieval, batch.person_ids)

        breakdown = stage2_loss_breakdown(
            model,
            batch,
            Stage2LossInputs(classifier=classifier, tfc_bank=tfc_bank, anchors=torch.eye(2)),
        )

        self.assertLess(float(breakdown.tfc_local.detach()), 0.01)
        self.assertGreaterEqual(float(breakdown.tfc.detach()), 0.0)


class Stage2ForwardLayoutTest(unittest.TestCase):
    def test_forward_stage2_returns_image_side_features_only(self):
        # The alignment term classifies against the precomputed identity anchor
        # matrix, so Stage-2 must not encode per-sample training text anymore.
        model = _training_model(beta=0.0)

        outputs = model.forward_stage2(torch.eye(2), torch.tensor([0, 1]), torch.tensor([0, 1]))

        self.assertEqual(set(outputs), {"visual_raw", "bn", "visual", "retrieval"})

    def test_encode_retrieval_validates_mode_before_running_the_image_tower(self):
        class CountingEncoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.calls = 0

            def forward(self, images, camera_ids=None):
                self.calls += 1
                return images

        encoder = CountingEncoder()
        model = _training_model(beta=0.0)
        model.image_encoder = encoder

        with self.assertRaises(ValueError):
            model.encode_retrieval(torch.eye(2), torch.tensor([0, 1]), retrieval_mode="bogus")

        self.assertEqual(encoder.calls, 0)

    def test_forward_stage2_beta_zero_retrieval_equals_l2n_bn(self):
        head = torch.nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            head.weight.copy_(torch.tensor([[2.0, 1.0], [0.0, 1.0]]))
        model = _training_model(beta=0.0, feature_head=head, camera_prompt=torch.tensor([[0.6, 0.8]]))
        images = torch.tensor([[1.0, 2.0], [3.0, -1.0]])
        camera_ids = torch.tensor([0, 1])
        person_ids = torch.tensor([0, 1])

        outputs = model.forward_stage2(images, camera_ids, person_ids)

        self.assertTrue(torch.equal(outputs["visual_raw"], images))
        self.assertTrue(torch.equal(outputs["bn"], head(images)))
        self.assertTrue(torch.equal(outputs["visual"], l2_normalize(images)))
        self.assertTrue(torch.equal(outputs["retrieval"], l2_normalize(outputs["bn"])))

    def test_encode_retrieval_fused_matches_training_retrieval_in_eval_mode(self):
        head = torch.nn.BatchNorm1d(2)
        with torch.no_grad():
            head.weight.copy_(torch.tensor([1.0, 3.0]))
            head.running_mean.copy_(torch.tensor([0.5, -0.5]))
            head.running_var.copy_(torch.tensor([4.0, 0.25]))
        model = _training_model(beta=0.7, feature_head=head, camera_prompt=torch.tensor([[0.6, 0.8]]))
        model.eval()
        images = torch.tensor([[1.0, 2.0], [3.0, -1.0]])
        camera_ids = torch.tensor([0, 1])

        training_retrieval = model.forward_stage2(images, camera_ids, torch.tensor([0, 1]))["retrieval"]
        eval_retrieval = model.encode_retrieval(images, camera_ids, retrieval_mode="fused")

        self.assertTrue(torch.equal(eval_retrieval, training_retrieval))

    def test_encode_retrieval_image_only_routes_through_feature_head(self):
        head = torch.nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            head.weight.copy_(torch.tensor([[2.0, 1.0], [0.0, 1.0]]))
        model = _training_model(beta=0.7, feature_head=head, camera_prompt=torch.tensor([[0.6, 0.8]]))
        images = torch.tensor([[1.0, 2.0], [3.0, -1.0]])

        output = model.encode_retrieval(images, torch.tensor([0, 1]), retrieval_mode="image_only")

        self.assertTrue(torch.equal(output, l2_normalize(head(images))))

    def test_feature_head_defaults_to_identity(self):
        model = _training_model(beta=0.0)

        self.assertIsInstance(model.feature_head, torch.nn.Identity)

    def test_encode_visual_raw_returns_unnormalized_feature(self):
        model = _training_model(beta=0.0)
        images = torch.tensor([[3.0, 4.0]])

        raw = model.encode_visual_raw(images)

        self.assertTrue(torch.equal(raw, images))
        self.assertTrue(torch.allclose(model.encode_visual(images), l2_normalize(images)))


class InferenceTextCacheTest(unittest.TestCase):
    def test_cached_inference_text_matches_online_encoding_bitwise(self):
        model = _training_model(beta=0.7, camera_prompt=torch.tensor([[0.6, 0.8]]))
        with torch.no_grad():
            model.prompt_bank.camera_prompts[1] = torch.tensor([[1.0, -2.0]])
        camera_ids = torch.tensor([1, 0, 1])
        online = model.encode_inference_text(camera_ids)

        cache = model.encode_inference_text(torch.arange(2))
        model.set_inference_text_cache(cache)
        cached = model.encode_inference_text(camera_ids)

        self.assertTrue(torch.equal(cached, online))

    def test_installed_cache_bypasses_the_text_encoder(self):
        model = _training_model(beta=0.0)
        cache = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        model.set_inference_text_cache(cache)
        with torch.no_grad():
            model.prompt_bank.camera_prompts.fill_(123.0)  # must not matter while cached

        output = model.encode_inference_text(torch.tensor([1, 1, 0]))

        self.assertTrue(torch.equal(output, cache[torch.tensor([1, 1, 0])]))

    def test_clearing_cache_restores_online_encoding(self):
        model = _training_model(beta=0.0, camera_prompt=torch.tensor([[0.6, 0.8]]))
        camera_ids = torch.tensor([0, 1])
        online = model.encode_inference_text(camera_ids)
        model.set_inference_text_cache(torch.zeros(2, 2))

        model.set_inference_text_cache(None)

        self.assertTrue(torch.equal(model.encode_inference_text(camera_ids), online))

    def test_set_inference_text_cache_validates_shape(self):
        model = _training_model(beta=0.0)

        with self.assertRaises(ValueError):
            model.set_inference_text_cache(torch.zeros(2))
        with self.assertRaises(ValueError):
            model.set_inference_text_cache(torch.zeros(3, 2))  # bank has 2 cameras

    def test_cached_encode_inference_text_validates_camera_ids(self):
        model = _training_model(beta=0.0)
        model.set_inference_text_cache(torch.zeros(2, 2))

        with self.assertRaises(ValueError):
            model.encode_inference_text(torch.tensor([0.5]))
        with self.assertRaises(ValueError):
            model.encode_inference_text(torch.tensor([2]))
        with self.assertRaises(ValueError):
            model.encode_inference_text(torch.tensor([-1]))


def _training_model(
    beta: float,
    feature_head: torch.nn.Module | None = None,
    camera_prompt: torch.Tensor | None = None,
) -> T2CReIDModel:
    bank = PromptBank(PromptConfig(num_cameras=2, num_train_ids=2, context_length=1, embedding_dim=2))
    with torch.no_grad():
        bank.global_prompt.zero_()
        bank.camera_prompts.zero_()
        if camera_prompt is not None:
            bank.camera_prompts.copy_(camera_prompt.unsqueeze(0).expand_as(bank.camera_prompts))
        bank.identity_prompts[0] = torch.tensor([[1.0, 0.0]])
        bank.identity_prompts[1] = torch.tensor([[0.0, 1.0]])
    return T2CReIDModel(IdentityEncoder(), PromptMeanEncoder(), bank, beta=beta, feature_head=feature_head)
