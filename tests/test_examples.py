import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from afno.data import generate
from afno.demo import load_sample, predict, verify_assets
from afno.models import AFNO
from afno.solvers import initial_conditions, solve
from afno.train import train
from afno.workflow import preset, run


class ExampleTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_afno_uses_batched_euler_step_and_encodes_initial_state_once(self):
        model = AFNO(width=4, modes=2, depth=1, latent_channels=2, embedding_dim=2)
        initial = torch.randn(2, 1, 16)
        params = torch.full((2, 1), .01)
        dt = torch.tensor([.01, .02])
        z = model.encoder(initial)
        expected = z + dt[:, None, None] * model.velocity(z, params, dt)
        torch.testing.assert_close(model.advance(z, params, dt), expected, rtol=0, atol=0)
        calls = []
        handle = model.encoder.register_forward_hook(lambda *args: calls.append(1))
        try:
            self.assertEqual(model.rollout(initial, params, dt, 7).shape, (2, 7, 1, 16))
        finally:
            handle.remove()
        self.assertEqual(len(calls), 1)

    def test_pretrained_inference_matches_saved_fields(self):
        self.assertTrue(verify_assets()["verified"])
        for case in ("burgers", "ks"):
            sample = load_sample(case)
            for name in ("fno", "afno"):
                with self.subTest(equation=case, model=name):
                    prediction = predict(case, name, sample)
                    # Float32 model inference can differ with batch size and BLAS/FFT backend.
                    np.testing.assert_allclose(prediction, sample[name + "_prediction"],
                                               rtol=2e-4, atol=2e-4)

    def test_reference_solvers_regenerate_bundled_validation_trajectory(self):
        for case in ("burgers", "ks"):
            config = preset(case)["data_generation"]
            seed = np.random.SeedSequence(config["split_seeds"]["val"]).spawn(1)[0]
            initial = initial_conditions(config, seed, 1)
            truth = solve(config, initial).astype(np.float32)
            sample = load_sample(case)
            np.testing.assert_allclose(truth[:, 0], sample["initial"], rtol=1e-6, atol=1e-6)
            np.testing.assert_allclose(truth[:, 1:], sample["target"], rtol=1e-6, atol=1e-6)

    def test_base_training_resume_is_exact(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(tmp)
            config = preset("smoke_burgers")
            config["data_path"] = str(root / "data")
            generate(config["data_generation"], config["data_path"], ["train", "val"])
            for name in ("fno", "afno"):
                cfg = copy.deepcopy(config)
                cfg["model"]["name"] = name
                train(cfg, root / f"{name}_continuous")
                train(cfg, root / f"{name}_resumed", stop_after=2)
                train(cfg, root / f"{name}_resumed", resume=True)
                a = torch.load(root / f"{name}_continuous/final.pt", weights_only=False)
                b = torch.load(root / f"{name}_resumed/final.pt", weights_only=False)
                for key in a["model"]:
                    torch.testing.assert_close(a["model"][key], b["model"][key], rtol=0, atol=0)
                self.assertEqual(a["scheduler"], b["scheduler"])

    def test_both_complete_workflows_resume_and_reject_changed_artifacts(self):
        with tempfile.TemporaryDirectory(prefix="afno examples ") as tmp, contextlib.redirect_stdout(io.StringIO()):
            for case in ("burgers", "ks"):
                root = Path(tmp) / case
                summary = run(case, root, smoke=True)
                self.assertTrue(summary["complete"])
                self.assertFalse((root / "data/test.npy").exists())
                before = (root / "afno/final.pt").read_bytes()
                run(case, root, smoke=True, resume=True)
                self.assertEqual((root / "afno/final.pt").read_bytes(), before)
                for name in ("FNO", "AFNO"):
                    self.assertEqual(summary["validation"][name]["steps"], 8)
                    self.assertEqual(summary["validation"][name]["nonfinite_trajectories"], 0)
                # A completed marker must not silently trust replaced checkpoints.
                with (root / "afno/final.pt").open("ab") as stream:
                    stream.write(b"changed")
                with self.assertRaisesRegex(ValueError, "changed after completion"):
                    run(case, root, smoke=True, resume=True)


if __name__ == "__main__":
    unittest.main()
