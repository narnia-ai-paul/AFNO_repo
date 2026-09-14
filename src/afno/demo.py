"""Pretrained inference using only the small assets distributed in this package."""
import json
from pathlib import Path

import numpy as np
import torch

from .data import atomic_json, file_hash, resize_numpy
from .models import build_model
from .train import seed_all
from .workflow import ASSETS


def verify_assets():
    manifest = json.loads((ASSETS / "manifest.json").read_text())
    for filename, expected in manifest["asset_sha256"].items():
        if file_hash(ASSETS / filename) != expected:
            raise ValueError(f"Bundled asset checksum mismatch: {filename}")
    return {"verified": True, "files": len(manifest["asset_sha256"]),
            "equations": list(manifest["cases"]), "split": manifest["split"]}


def load_sample(equation):
    if equation not in ("burgers", "ks"):
        raise ValueError("Choose burgers or ks")
    with np.load(ASSETS / "examples" / f"{equation}_sample.npz", allow_pickle=False) as archive:
        return {k: archive[k] for k in archive.files}


def load_model(equation, name, device="cpu"):
    if equation not in ("burgers", "ks") or name not in ("fno", "afno"):
        raise ValueError("Unknown example or model")
    # Distributed files contain only tensors and primitive metadata, no optimizer/RNG pickle objects.
    state = torch.load(ASSETS / "pretrained" / f"{equation}_{name}.pt",
                       map_location=device, weights_only=True)
    model = build_model(state["model_config"]).to(device).eval()
    model.load_state_dict(state["state_dict"], strict=True)
    return model, state


@torch.no_grad()
def predict(equation, name, sample, device="cpu"):
    model, state = load_model(equation, name, device)
    cfg = state["data_generation"]
    initial = torch.from_numpy(resize_numpy(sample["initial"], state["training_resolution"])).to(device)
    params = torch.tensor([[cfg["params"][key] for key in cfg["conditioning"]]],
                           dtype=torch.float32, device=device).expand(len(initial), -1)
    dt = torch.full((len(initial),), cfg["observation_dt"], device=device)
    low = model.rollout(initial, params, dt, len(sample["time"]))
    prediction = resize_numpy(low.cpu().numpy(), [len(sample["x"])])
    if not np.isfinite(prediction).all():
        raise FloatingPointError(f"Nonfinite {equation} {name} demo prediction")
    return prediction


def run(equation="both", output="outputs/demo", device="cpu", figures=True):
    verified = verify_assets()
    seed_all(0)
    torch.set_num_threads(2)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    cases = ("burgers", "ks") if equation == "both" else (equation,)
    bundles, result = {}, {"assets": verified, "device": device, "split": "val",
                          "sample_index": 0, "trajectories_per_equation": 1,
                          "note": "Sample metrics only; full-population results are bundled separately.",
                          "cases": {}}
    for case in cases:
        sample = load_sample(case)
        bundles[case], result["cases"][case] = {}, {}
        for name in ("fno", "afno"):
            prediction = predict(case, name, sample, device)
            saved = sample[name + "_prediction"]
            mse = ((prediction.astype(np.float64) - sample["target"].astype(np.float64))**2).mean((2, 3))
            delta = np.abs(prediction.astype(np.float64) - saved.astype(np.float64))
            result["cases"][case][name.upper()] = {
                "first5_mse": float(mse[:, :5].mean()), "full100_mse": float(mse.mean()),
                "max_abs_difference_from_saved_prediction": float(delta.max())}
            bundles[case][name] = {k: sample[k] for k in ("initial", "target", "indices", "time", "x")}
            bundles[case][name].update(prediction=prediction, mse=mse)
            np.savez_compressed(output / f"{case}_{name}.npz", **bundles[case][name])
            print(f"{case} {name.upper()}: first-five MSE={mse[:, :5].mean():.6g}, "
                  f"full-100 MSE={mse.mean():.6g}, difference from saved fields={delta.max():.3g}", flush=True)
    atomic_json(output / "metrics.json", result)
    if figures:
        from .figures import render
        render(output, bundles=bundles, include_population=False)
    return result
