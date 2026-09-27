import unittest

import torch

from t2c_reid.clothes import (
    CLOTHES_LOGIT_SCALE,
    ClothesAdversarialSchedule,
    ClothesClassifier,
    build_clothes_table,
    clothes_adversarial_loss,
)


def _reference_adversarial_loss(logits, clothes_ids, positive_mask, epsilon):
    # Official Simple-CCReID ClothesBasedAdversarialLoss on pre-scaled logits.
    negative_mask = 1 - positive_mask
    identity_mask = torch.nn.functional.one_hot(clothes_ids, logits.shape[1]).float()
    exp_logits = torch.exp(logits)
    log_sum = torch.log((exp_logits * negative_mask).sum(1, keepdim=True) + exp_logits)
    log_prob = logits - log_sum
    mask = (1 - epsilon) * identity_mask + epsilon / positive_mask.sum(
        1, keepdim=True
    ) * positive_mask
    return (-mask * log_prob).sum(1).mean()


class ClothesTableTest(unittest.TestCase):
    def test_groups_cameras_per_identity_and_skips_unobserved_pairs(self):
        counts = torch.tensor([[2, 1, 3], [1, 0, 2], [0, 4, 0]])

        table = build_clothes_table(counts, [0, 0, 1])

        expected = torch.tensor([[0, 0, 1], [2, -1, 3], [-1, 4, -1]])
        self.assertTrue(torch.equal(table, expected))

    def test_rejects_mismatched_camera_groups_and_single_identity(self):
        with self.assertRaisesRegex(ValueError, "camera_groups"):
            build_clothes_table(torch.ones(2, 3, dtype=torch.long), [0, 1])
        with self.assertRaisesRegex(ValueError, "two identities"):
            build_clothes_table(torch.ones(1, 2, dtype=torch.long), [0, 1])


class ClothesAdversarialLossTest(unittest.TestCase):
    def test_matches_official_formula(self):
        generator = torch.Generator().manual_seed(0)
        logits = torch.randn(4, 5, generator=generator) * 4.0
        clothes_ids = torch.tensor([0, 1, 2, 4])
        positive_mask = torch.tensor(
            [
                [1, 1, 0, 0, 0],
                [1, 1, 0, 0, 0],
                [0, 0, 1, 1, 0],
                [0, 0, 0, 0, 1],
            ],
            dtype=torch.float32,
        )

        loss = clothes_adversarial_loss(logits, clothes_ids, positive_mask, 0.1)

        expected = _reference_adversarial_loss(logits, clothes_ids, positive_mask, 0.1)
        torch.testing.assert_close(loss, expected)

    def test_rejects_target_outside_positive_mask(self):
        positive_mask = torch.tensor([[1, 0], [0, 1]], dtype=torch.float32)

        with self.assertRaisesRegex(ValueError, "positive"):
            clothes_adversarial_loss(torch.zeros(2, 2), torch.tensor([1, 1]), positive_mask)


class ClothesClassifierTest(unittest.TestCase):
    def _classifier(self) -> ClothesClassifier:
        torch.manual_seed(0)
        table = build_clothes_table(torch.ones(2, 3, dtype=torch.long), [0, 0, 1])
        return ClothesClassifier(feature_dim=4, clothes_table=table)

    def test_classifier_and_adversarial_gradients_are_disjoint(self):
        classifier = self._classifier()
        person_ids = torch.tensor([0, 0, 1, 1])
        camera_ids = torch.tensor([0, 2, 1, 2])

        features = torch.randn(4, 4, requires_grad=True)
        classifier.loss(features, person_ids, camera_ids).classifier.backward()
        self.assertIsNone(features.grad)
        self.assertGreater(float(classifier.weight.grad.abs().sum()), 0.0)

        classifier.zero_grad(set_to_none=True)
        features = torch.randn(4, 4, requires_grad=True)
        classifier.loss(features, person_ids, camera_ids).adversarial.backward()
        self.assertIsNone(classifier.weight.grad)
        self.assertGreater(float(features.grad.abs().sum()), 0.0)

    def test_labels_positive_mask_and_fp32_logits_under_autocast(self):
        classifier = self._classifier()

        clothes_ids = classifier.clothes_ids(torch.tensor([0, 0, 1]), torch.tensor([1, 2, 2]))
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            logits = classifier.logits(torch.randn(3, 4), detach_weight=False)

        self.assertEqual(clothes_ids.tolist(), [0, 1, 3])
        self.assertEqual(classifier.positive_mask.tolist(), [[1, 1, 0, 0], [0, 0, 1, 1]])
        self.assertEqual(logits.dtype, torch.float32)
        self.assertLessEqual(float(logits.abs().max()), CLOTHES_LOGIT_SCALE + 1e-4)
        self.assertNotIn("clothes_table", classifier.state_dict())

    def test_rejects_unobserved_pid_camera_pair(self):
        counts = torch.tensor([[1, 0, 1], [1, 1, 1]])
        classifier = ClothesClassifier(4, build_clothes_table(counts, [0, 0, 1]))

        with self.assertRaisesRegex(ValueError, "absent"):
            classifier.clothes_ids(torch.tensor([0]), torch.tensor([1]))


class ClothesAdversarialScheduleTest(unittest.TestCase):
    def test_enables_weight_from_stage_local_start_epoch(self):
        table = build_clothes_table(torch.ones(2, 2, dtype=torch.long), [0, 1])
        classifier = ClothesClassifier(4, table)
        schedule = ClothesAdversarialSchedule(weight=0.5, start_epoch=2, first_epoch=61)

        schedule.apply(classifier, 61)
        self.assertEqual(classifier.adversarial_weight, 0.0)
        schedule.apply(classifier, 62)
        self.assertEqual(classifier.adversarial_weight, 0.5)


if __name__ == "__main__":
    unittest.main()
