import unittest

import torch

from t2c_reid.features import fuse_features, l2_normalize
from t2c_reid.prompts import PromptBank, PromptConfig


class FeaturePromptTest(unittest.TestCase):
    def test_l2_normalize_returns_unit_rows(self):
        output = l2_normalize(torch.tensor([[3.0, 4.0]]))
        self.assertTrue(torch.allclose(output.norm(dim=1), torch.ones(1)))

    def test_fuse_features_uses_beta_and_normalizes(self):
        visual = torch.tensor([[1.0, 0.0]])
        text = torch.tensor([[0.0, 1.0]])
        fused = fuse_features(visual, text, beta=1.0)
        expected = torch.tensor([[2**-0.5, 2**-0.5]])
        self.assertTrue(torch.allclose(fused, expected))

    def test_fuse_features_beta_zero_returns_l2_normalized_visual_bitwise(self):
        # beta=0 must yield the pure image-side retrieval feature EXACTLY:
        # double-normalization (l2n(l2n(v) + 0)) is not bitwise idempotent.
        torch.manual_seed(0)
        visual = torch.randn(8, 16)
        text = torch.randn(8, 16)

        fused = fuse_features(visual, text, beta=0.0)

        self.assertTrue(torch.equal(fused, l2_normalize(visual)))

    def test_fusion_beta_is_independent_of_input_norms(self):
        visual = torch.tensor([[100.0, 0.0]])
        text = torch.tensor([[0.0, 7.0]])
        expected = l2_normalize(torch.tensor([[1.0, 0.1]]))
        torch.testing.assert_close(fuse_features(visual, text, beta=0.1), expected)

    def test_prompt_bank_training_adds_identity_prompt(self):
        bank = PromptBank(
            PromptConfig(
                num_cameras=2, num_train_ids=3, context_length=2, embedding_dim=2
            )
        )
        with torch.no_grad():
            bank.global_prompt.fill_(1.0)
            bank.camera_prompts.zero_()
            bank.identity_prompts.zero_()
            bank.camera_prompts[1].fill_(2.0)
            bank.identity_prompts[2].fill_(4.0)

        prompt = bank.training_prompts(torch.tensor([1]), torch.tensor([2]))

        self.assertTrue(torch.equal(prompt, torch.full((1, 2, 2), 7.0)))

    def test_prompt_bank_inference_excludes_identity_prompt(self):
        bank = PromptBank(
            PromptConfig(
                num_cameras=2, num_train_ids=3, context_length=1, embedding_dim=2
            )
        )
        with torch.no_grad():
            bank.global_prompt.fill_(1.0)
            bank.camera_prompts[1].fill_(2.0)
            bank.identity_prompts[2].fill_(4.0)

        prompt = bank.inference_prompts(torch.tensor([1]))

        self.assertTrue(torch.equal(prompt, torch.full((1, 1, 2), 3.0)))

    def test_prompt_bank_identity_anchor_excludes_camera_prompt(self):
        # The alignment anchor is a camera-agnostic identity prototype: global +
        # identity only, never the camera component.
        bank = PromptBank(
            PromptConfig(
                num_cameras=2, num_train_ids=3, context_length=2, embedding_dim=2
            )
        )
        with torch.no_grad():
            bank.global_prompt.fill_(1.0)
            bank.camera_prompts.fill_(2.0)
            bank.identity_prompts.zero_()
            bank.identity_prompts[2].fill_(4.0)

        prompt = bank.identity_anchor_prompts(torch.tensor([2]))

        self.assertTrue(torch.equal(prompt, torch.full((1, 2, 2), 5.0)))

    def test_prompt_bank_identity_anchor_validates_person_ids(self):
        bank = PromptBank(
            PromptConfig(
                num_cameras=2, num_train_ids=3, context_length=1, embedding_dim=2
            )
        )

        with self.assertRaises(ValueError):
            bank.identity_anchor_prompts(torch.tensor([0.5]))
        with self.assertRaises(ValueError):
            bank.identity_anchor_prompts(torch.tensor([3]))
        with self.assertRaises(ValueError):
            bank.identity_anchor_prompts(torch.tensor([-1]))
