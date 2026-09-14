"""Freeze TRAIN-only loss scales; independently verify the FNO validation reference."""
import argparse
import json
from pathlib import Path

import torch

from afno.data import atomic_json
from .evaluation import checked_data, evaluate, initialize
from .balanced_objectives import alpha_at, rollout_loss


def calibrate(run, output, fno_reference, study, device):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    model, config, _, identity = initialize(run, device)
    data = checked_data(config, "train", config["training_resolution"])
    metrics, _ = evaluate(model, data, device, study["training"]["batch_size"], probe_count=0)
    if metrics["nonfinite_trajectories"]:
        raise FloatingPointError("Nonfinite calibration; never remove failed training trajectories")
    if metrics["split"] != "train":
        raise ValueError("Calibration requires training predictions")
    scales = {"early": metrics["first5_mse"], "full": metrics["mean_mse"]}
    if min(scales.values()) <= 0:
        raise ValueError("Zero reference error; explicit alternative scaling would be needed")
    reference = {"initial_checkpoint_sha256": identity,
                 "train_sha256": data.manifest["splits"]["train"]["sha256"],
                 "train_trajectories": len(data), "train_steps": data.meta["steps"],
                 "train_scales": scales, "fno_validation": fno_reference,
                 "scaling_note": "Fixed mean MSEs of the initial model on ALL TRAIN trajectories, no validation fitting."}
    atomic_json(output/"references.json", reference)
    # The first real batch checks each declared objective at full training size.
    u, p, dt = next(iter(torch.utils.data.DataLoader(data, batch_size=study["training"]["batch_size"])))
    u, p, dt = u.to(device), p.to(device), dt.to(device)
    gate = []
    for arm in study["arms"]:
        model.train(); model.zero_grad(set_to_none=True)
        if device == "cuda": torch.cuda.reset_peak_memory_stats()
        loss, _ = rollout_loss(model, u, p, dt, data.meta["lengths"], scales, alpha_at(1, arm), arm["mode"],
                               study["training"]["rec_weight"], study["training"]["h1_weight"],
                               study["training"]["activation_checkpoint"])
        if not torch.isfinite(loss): raise FloatingPointError("Nonfinite gate loss")
        loss.backward()
        if not all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()):
            raise FloatingPointError("Missing/nonfinite gate gradient")
        gate.append({"arm": arm["name"], "loss": loss.item(), "batch_size": len(u),
                     "horizon": u.shape[1]-1, "all_gradients_finite": True,
                     "peak_cuda_bytes": torch.cuda.max_memory_allocated() if device == "cuda" else None})
    atomic_json(output/"gate.json", {"passed": True, "cases": gate})
    print(json.dumps(reference, allow_nan=False))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", required=True); p.add_argument("--output", required=True)
    p.add_argument("--study", required=True); p.add_argument("--fno-reference", required=True)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    a = p.parse_args()
    calibrate(a.run, a.output, json.loads(Path(a.fno_reference).read_text()),
              json.loads(Path(a.study).read_text()), a.device)
