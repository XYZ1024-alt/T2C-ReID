import unittest

import torch

from t2c_reid.ema import ModelEma


class ModelEmaTest(unittest.TestCase):
    def test_rejects_decay_outside_open_unit_interval(self):
        for decay in (0.0, 1.0, -0.5, float("nan")):
            with self.subTest(decay=decay):
                with self.assertRaisesRegex(ValueError, "0 < decay < 1"):
                    ModelEma(torch.nn.Linear(2, 2), decay)

    def test_tracks_trainable_parameters_and_floating_buffers_only(self):
        module = _TinyModule()
        module.frozen.requires_grad_(False)

        ema = ModelEma(module, 0.9)
        ema.initialize()

        self.assertIn("linear.weight", ema.tracked_names)
        self.assertIn("norm.running_mean", ema.tracked_names)
        self.assertNotIn("frozen.weight", ema.tracked_names)
        self.assertNotIn("norm.num_batches_tracked", ema.tracked_names)

    def test_update_averages_during_warmup_then_decays_exponentially(self):
        module = torch.nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            module.weight.fill_(-100.0)
        ema = ModelEma(module, 0.75)
        ema.initialize()

        # Warmup keeps the plain mean of the updated weights and drops the
        # initialization snapshot: mean(2, 4, 6) = 4. From the fourth update
        # on decay_t reaches 0.75: 0.75 * 4 + 0.25 * 8 = 5, then 6.25.
        expected = (2.0, 3.0, 4.0, 5.0, 6.25)
        for live, shadow in zip((2.0, 4.0, 6.0, 8.0, 10.0), expected):
            with torch.no_grad():
                module.weight.fill_(live)
            ema.update()
            self.assertAlmostEqual(
                ema.state_dict()["shadow"]["weight"].item(), shadow, places=5
            )
        self.assertEqual(ema.num_updates, 5)
        self.assertEqual(module.weight.item(), 10.0)

    def test_applied_swaps_shadow_in_and_restores_live_weights(self):
        module = torch.nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            module.weight.fill_(1.0)
        ema = ModelEma(module, 0.5)
        ema.initialize()
        for live in (3.0, 5.0):
            with torch.no_grad():
                module.weight.fill_(live)
            ema.update()

        with ema.applied():
            self.assertEqual(module.weight.item(), 4.0)
        self.assertEqual(module.weight.item(), 5.0)

        with self.assertRaises(RuntimeError):
            with ema.applied():
                raise RuntimeError("evaluation failed")
        self.assertEqual(module.weight.item(), 5.0)

    def test_use_before_initialize_fails(self):
        ema = ModelEma(torch.nn.Linear(1, 1), 0.5)

        with self.assertRaisesRegex(ValueError, "before it was initialized"):
            ema.update()
        with self.assertRaisesRegex(ValueError, "before it was initialized"):
            with ema.applied():
                pass

    def test_state_round_trip_restores_shadow_on_fresh_instance(self):
        module = _TinyModule()
        source = ModelEma(module, 0.5)
        source.initialize()
        with torch.no_grad():
            module.linear.weight.add_(1.0)
        source.update()
        state = source.state_dict()

        restored = ModelEma(module, 0.5)
        restored.load_state_dict(state)

        self.assertTrue(restored.initialized)
        self.assertEqual(restored.tracked_names, source.tracked_names)
        self.assertEqual(restored.num_updates, 1)
        for name, tensor in restored.state_dict()["shadow"].items():
            torch.testing.assert_close(tensor, state["shadow"][name])
            self.assertIsNot(tensor, state["shadow"][name])

    def test_load_rejects_decay_or_shape_mismatch(self):
        module = torch.nn.Linear(2, 2)
        source = ModelEma(module, 0.5)
        source.initialize()
        state = source.state_dict()

        with self.assertRaisesRegex(ValueError, "decay does not match"):
            ModelEma(module, 0.9).load_state_dict(state)
        with self.assertRaisesRegex(ValueError, "does not match the model shape"):
            ModelEma(torch.nn.Linear(3, 2), 0.5).load_state_dict(state)
        with self.assertRaisesRegex(ValueError, "missing from the model"):
            ModelEma(torch.nn.Sequential(torch.nn.Linear(2, 2)), 0.5).load_state_dict(
                state
            )

    def test_load_of_uninitialized_state_stays_uninitialized(self):
        module = torch.nn.Linear(1, 1)
        state = ModelEma(module, 0.5).state_dict()

        restored = ModelEma(module, 0.5)
        restored.load_state_dict(state)

        self.assertFalse(restored.initialized)


class _TinyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
        self.norm = torch.nn.BatchNorm1d(2)
        self.frozen = torch.nn.Linear(2, 2, bias=False)


if __name__ == "__main__":
    unittest.main()
