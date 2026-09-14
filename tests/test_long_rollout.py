import contextlib
import copy
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch

from afno.data import generate, resize_numpy
from afno.losses import training_loss
from afno.models import AFNO, FNO
from afno.train import train as paper_train

from afno.workflow import preset
from afno.evaluation import checked_data, evaluate, initialize
from afno.warmup_objectives import block_ranges, eligible_score, horizon_at, rollout_loss
from afno import warmup as module


class LongRolloutTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(13)
        torch.set_num_threads(2)

    def test_five_step_control_matches_existing_loss_and_gradients(self):
        a = AFNO(width=4, depth=1, modes=2, latent_channels=2, embedding_dim=2)
        b = copy.deepcopy(a)
        u = torch.randn(2, 6, 1, 16); p = torch.full((2, 1), .01); dt = torch.full((2,), .01)
        x, _ = training_loss(a, u, p, dt, [1.], [0., .5, .1])
        y, _ = rollout_loss(b, u, p, dt, 0., [1.], 1e-4, True)
        x.backward(); y.backward()
        torch.testing.assert_close(x, y, rtol=1e-6, atol=1e-7)
        for pa, pb in zip(a.parameters(), b.parameters()):
            torch.testing.assert_close(pa.grad, pb.grad, rtol=1e-5, atol=1e-6)

    def test_checkpointed_long_rollout_preserves_all_gradients_and_target_indices(self):
        a = AFNO(width=4, depth=1, modes=2, latent_channels=2, embedding_dim=2)
        b = copy.deepcopy(a)
        u = torch.randn(2, 26, 1, 16); p = torch.full((2, 1), .01); dt = torch.tensor([.01, .02])
        torch.manual_seed(7)
        x, _ = rollout_loss(a, u, p, dt, .4, [1.], 1e-4, False, True)
        torch.manual_seed(7)
        y, _ = rollout_loss(b, u, p, dt, .4, [1.], 1e-4, True, True)
        x.backward(); y.backward()
        torch.testing.assert_close(x, y, rtol=0, atol=0)
        for pa, pb in zip(a.parameters(), b.parameters()):
            self.assertIsNotNone(pa.grad)
            torch.testing.assert_close(pa.grad, pb.grad, rtol=0, atol=0)

    def test_step_100_target_gradient_reaches_initial_encoder_analytically(self):
        class Gain(torch.nn.Module):
            def __init__(self):
                super().__init__(); self.weight = torch.nn.Parameter(torch.tensor(1., dtype=torch.float64))
            def forward(self, x): return self.weight*x
        class LinearODE(torch.nn.Module):
            ndim = 1
            def __init__(self):
                super().__init__(); self.encoder = Gain(); self.decoder = torch.nn.Identity()
            def velocity(self, z, p, dt): return .2*z
            def advance(self, z, p, dt): return z + dt[:, None, None]*self.velocity(z, p, dt)
        model = LinearODE()
        states = torch.zeros(1, 101, 1, 1, dtype=torch.float64); states[:, 0] = 2.
        params = torch.zeros(1, 1, dtype=torch.float64); dt = torch.tensor([.1], dtype=torch.float64)
        before, _ = rollout_loss(model, states, params, dt, 0., [1.], 0., True)
        g0, = torch.autograd.grad(before, model.encoder.weight)
        states[:, -1] = 3.
        after, _ = rollout_loss(model, states, params, dt, 0., [1.], 0., True)
        g1, = torch.autograd.grad(after, model.encoder.weight)
        # Only the last target changed. Its exact contribution to dL/dw is
        # -(delta_target/K) * u0 * (1 + dt*rate)^K.
        expected = -3./100*2.*1.02**100
        self.assertAlmostEqual((g1-g0).item(), expected, places=11)

    def test_selection_rejects_early_regression_and_nonfinite(self):
        initial = {"first5_mse": 1.}
        metrics = {"first5_mse": 1.11, "nonfinite_trajectories": 0, "blocks": [{"mse": .2}]}
        self.assertIsNone(eligible_score(metrics, initial, .1))
        metrics["first5_mse"] = 1.09
        self.assertEqual(eligible_score(metrics, initial, .1), .2)
        metrics["nonfinite_trajectories"] = 1
        self.assertIsNone(eligible_score(metrics, initial, .1))
        self.assertEqual(list(block_ranges(100)), [(0, 5), (5, 20), (20, 50), (50, 100)])

    def test_validation_resamples_readout_and_includes_every_step(self):
        class Data(torch.utils.data.Dataset):
            resolution = [32]
            meta = {"equation": "burgers"}
            manifest = {"splits": {"val": {"sha256": "fixture"}}}
            def __init__(self):
                self.states = torch.randn(3, 8, 1, 32)
            def __len__(self): return len(self.states)
            def __getitem__(self, i):
                return self.states[i], torch.tensor([.01]), torch.tensor(.01)
        data = Data()
        model = FNO(width=4, depth=1, modes=2)
        result, arrays = evaluate(model, data, "cpu", 2, training_resolution=[16])
        with torch.no_grad():
            initial = torch.from_numpy(resize_numpy(data.states[:, 0].numpy(), [16]))
            low = model.rollout(initial, torch.full((3, 1), .01), .01, 7)
            pred = resize_numpy(low.numpy(), [32])
        expected = ((pred.astype('float64')-data.states[:, 1:].numpy())**2).mean((2, 3))
        np.testing.assert_allclose(arrays["mse"], expected, rtol=1e-6, atol=1e-7)
        self.assertEqual(result["trajectories"], 3); self.assertEqual(result["steps"], 7)

    def test_curriculum_resume_exact_and_no_test_file(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(tmp)
            config = preset("smoke_burgers")
            config["data_path"] = str(root/"data")
            generate(config["data_generation"], config["data_path"], split_names=["train", "val"])
            paper_train(config, root/"initial")
            study = preset("warmup")
            study["training"].update(epochs=4, batch_size=4, validate_every=2)
            arm = {"name": "fixture", "rec_weight": .4, "balance_blocks": True,
                   "stages": [{"through_epoch": 2, "horizon": 5}, {"through_epoch": 4, "horizon": 8}]}
            self.assertEqual([horizon_at(i, arm["stages"]) for i in range(1, 5)], [5, 5, 8, 8])
            module.train(study, arm, root/"initial", root/"continuous")
            module.train(study, arm, root/"initial", root/"resumed", stop_after=2)
            module.train(study, arm, root/"initial", root/"resumed")
            a = torch.load(root/"continuous/final.pt", weights_only=False)
            b = torch.load(root/"resumed/final.pt", weights_only=False)
            for key in a["model"]:
                torch.testing.assert_close(a["model"][key], b["model"][key], rtol=0, atol=0)
            self.assertEqual(a["scheduler"], b["scheduler"])
            self.assertEqual([r["loss"] for r in a["history"]], [r["loss"] for r in b["history"]])
            self.assertEqual(a["best_epoch"], b["best_epoch"])
            altered = copy.deepcopy(study); altered["training"]["epochs"] = 5
            with self.assertRaises(ValueError):
                module.train(altered, arm, root/"initial", root/"resumed")
            with self.assertRaises(ValueError):
                checked_data(config, "test", [32])
            self.assertFalse((root/"data/test.npy").exists())


if __name__ == "__main__":
    unittest.main()
