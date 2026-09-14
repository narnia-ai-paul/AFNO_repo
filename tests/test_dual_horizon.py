import contextlib
import copy
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

import torch

from afno.data import generate
from afno.models import AFNO, FNO
from afno.train import train as paper_train

from afno.workflow import preset
from afno.evaluation import checked_data, evaluate, initialize
from afno.balanced_objectives import alpha_at, dual_score, rollout_loss
from afno.balanced import train
from afno.calibration import calibrate


class DualHorizonTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(2); torch.set_num_threads(2)

    def test_joint_score_cannot_hide_one_failed_horizon(self):
        reference = {"first5_mse": .01, "mean_mse": 1.}
        metrics = {"first5_mse": .02, "mean_mse": .1, "nonfinite_trajectories": 0}
        self.assertEqual(dual_score(metrics, reference), 2.)
        metrics["first5_mse"] = .005
        self.assertEqual(dual_score(metrics, reference), .5)
        metrics["nonfinite_trajectories"] = 1
        self.assertIsNone(dual_score(metrics, reference))

    def test_normalized_loss_matches_independent_rollout_and_keeps_gradients(self):
        model = AFNO(width=4, depth=1, modes=2, latent_channels=2, embedding_dim=2)
        plain = copy.deepcopy(model)
        u = torch.randn(2, 9, 1, 16); p = torch.ones(2, 1)*.01; dt = torch.tensor([.01, .02])
        scales = {"early": .01, "full": .3}; alpha = .7
        prediction = plain.rollout(u[:, 0], p, dt, 8)
        mse = (prediction-u[:, 1:]).square().flatten(2).mean(2)
        expected = scales["full"]*(alpha*mse[:, :5].mean()/scales["early"]+(1-alpha)*mse.mean()/scales["full"])
        loss, terms = rollout_loss(model, u, p, dt, [1.], scales, alpha, rec_weight=0.)
        torch.testing.assert_close(terms["weighted_roll"], expected.detach())
        direct, _ = rollout_loss(plain, u, p, dt, [1.], scales, alpha, rec_weight=0., activation_checkpoint=False)
        loss.backward(); direct.backward()
        for a, b in zip(model.parameters(), plain.parameters()):
            self.assertIsNotNone(a.grad)
            torch.testing.assert_close(a.grad, b.grad, rtol=0, atol=0)
        with self.assertRaises(ValueError):
            rollout_loss(model, u, p, dt, [1.], {"early": 0., "full": 1.}, alpha)

    def test_late_target_gradient_reaches_encoder_under_early_priority(self):
        class Gain(torch.nn.Module):
            def __init__(self):
                super().__init__(); self.weight = torch.nn.Parameter(torch.tensor(1., dtype=torch.float64))
            def forward(self, x): return x*self.weight
        class ODE(torch.nn.Module):
            ndim = 1
            def __init__(self):
                super().__init__(); self.encoder = Gain(); self.decoder = torch.nn.Identity()
            def velocity(self, z, p, dt): return .2*z
            def advance(self, z, p, dt): return z + dt[:, None, None]*self.velocity(z, p, dt)
        model = ODE(); u = torch.zeros(1, 101, 1, 1, dtype=torch.float64); u[:, 0] = 2.
        p = torch.zeros(1, 1, dtype=torch.float64); dt = torch.tensor([.1], dtype=torch.float64)
        scales = {"early": .001, "full": 1.}
        loss, _ = rollout_loss(model, u, p, dt, [1.], scales, .9, rec_weight=0.)
        g0, = torch.autograd.grad(loss, model.encoder.weight)
        u[:, -1] = 3.
        loss, _ = rollout_loss(model, u, p, dt, [1.], scales, .9, rec_weight=0.)
        g1, = torch.autograd.grad(loss, model.encoder.weight)
        self.assertAlmostEqual((g1-g0).item(), -.1*3./100*2.*1.02**100, places=9)

    def test_train_calibration_resume_and_test_firewall(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(tmp)
            config = preset("smoke_burgers")
            config["data_path"] = str(root/"data")
            generate(config["data_generation"], config["data_path"], split_names=["train", "val"])
            paper_train(config, root/"initial")
            study = preset("balanced")
            study["training"].update(epochs=4, horizon=8, batch_size=4, validate_every=2)
            arm = {"name": "fixture", "mode": "normalized", "stages": [
                {"through_epoch": 2, "early_weight": .9}, {"through_epoch": 4, "early_weight": .5}]}
            study["arms"] = [arm]
            model, _, _, _ = initialize(root/"initial", "cpu")
            val = checked_data(config, "val", [32]); comparator, _ = evaluate(model, val, "cpu", 4)
            calibrate(root/"initial", root/"calibration", comparator, study, "cpu")
            references = json.loads((root/"calibration/references.json").read_text())
            training = checked_data(config, "train", [32]); train_metrics, _ = evaluate(model, training, "cpu", 4)
            self.assertEqual(train_metrics["split"], "train")
            self.assertEqual(references["train_sha256"], train_metrics["data_sha256"])
            self.assertEqual(references["train_scales"]["early"], train_metrics["first5_mse"])
            self.assertEqual(references["train_trajectories"], 8)
            self.assertEqual([alpha_at(i, arm) for i in range(1, 5)], [.9, .9, .5, .5])
            train(study, arm, root/"initial", root/"continuous", references)
            train(study, arm, root/"initial", root/"resumed", references, stop_after=2)
            train(study, arm, root/"initial", root/"resumed", references)
            a = torch.load(root/"continuous/final.pt", weights_only=False)
            b = torch.load(root/"resumed/final.pt", weights_only=False)
            for k in a["model"]: torch.testing.assert_close(a["model"][k], b["model"][k], rtol=0, atol=0)
            self.assertEqual(a["scheduler"], b["scheduler"])
            self.assertEqual(a["best_epoch"], b["best_epoch"])
            self.assertEqual([r["loss"] for r in a["history"]], [r["loss"] for r in b["history"]])
            # The new checkpoint format remains loadable with the original model source.
            initialize(root/"continuous", "cpu")
            altered = copy.deepcopy(study); altered["training"]["epochs"] = 5
            with self.assertRaises(ValueError): train(altered, arm, root/"initial", root/"resumed", references)
            with self.assertRaises(ValueError): checked_data(config, "test", [32])
            self.assertFalse((root/"data/test.npy").exists())


if __name__ == "__main__":
    unittest.main()
