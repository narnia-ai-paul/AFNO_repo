"""Full-trajectory VALIDATION evaluation; no test split is accepted or opened."""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from afno.data import TrajectoryDataset, atomic_json, file_hash, resize_numpy
from afno.models import AFNO, build_model
from afno.train import seed_all, source_hash
from .balanced_objectives import block_ranges


def checked_data(config, split, resolution, horizon=None, seed=0):
    if split not in ("train", "val"):
        raise ValueError("Exploratory study permits train and val only")
    data = TrajectoryDataset(config["data_path"], split, resolution, horizon, seed)
    data.split = split
    entry = data.manifest["splits"][split]
    if file_hash(Path(config["data_path"])/entry["path"]) != entry["sha256"]:
        raise ValueError(f"{split} data hash mismatch")
    if data.meta != config["data_generation"]:
        raise ValueError("Dataset configuration mismatch")
    return data


def initialize(run, device):
    seed_all(0)
    torch.set_num_threads(int(os.getenv("OMP_NUM_THREADS", "2")))
    path = Path(run)/"final.pt"
    state = torch.load(path, map_location=device, weights_only=False)
    provenance = state["provenance"]
    initial_source = provenance.get("source_sha256", provenance.get("initial_source_sha256"))
    if initial_source != source_hash():
        raise ValueError("Model source changed; use the source version that created this run")
    config = provenance.get("config", provenance.get("base_config"))
    model = build_model(config["model"]).to(device)
    model.load_state_dict(state["model"], strict=True)
    return model, config, state, file_hash(path)


@torch.no_grad()
def evaluate(model, data, device, batch_size=8, limit=None, probe_count=8, training_resolution=None):
    """Keep all per-trajectory/per-step errors, including nonfinite failures.

    Relative L2 divides each sample/step by that target's physical-field norm.
    The absolute MSE is always reported too, including for decaying solutions.
    """
    model.eval()
    count = min(limit, len(data)) if limit else len(data)
    loader = DataLoader(Subset(data, range(count)), batch_size=batch_size, shuffle=False)
    values = {k: [] for k in ("mse", "relative_l2", "power", "centered_power")}
    probes = {k: [] for k in ("rec_mse", "latent_rmse", "decoder_secant_gain")}
    seen = 0
    for u, p, dt in loader:
        model_resolution = training_resolution or data.resolution
        needs_resize = list(model_resolution) != list(data.resolution)
        low_u = torch.from_numpy(resize_numpy(u.numpy(), model_resolution)).to(device) if needs_resize else u.to(device)
        u, p, dt = u.to(device), p.to(device), dt.to(device)
        is_afno = isinstance(model, AFNO)
        z = model.encoder(low_u[:, 0]) if is_afno else None
        field = low_u[:, 0]
        batch = {k: [] for k in values}
        probe = {k: [] for k in probes}
        probe_n = max(0, min(len(u), probe_count-seen)) if is_afno else 0
        for step in range(1, u.shape[1]):
            if is_afno:
                z = model.advance(z, p, dt)
                field = model.decoder(z)
            else:
                field = model.step(field, p, dt)
            readout = torch.from_numpy(resize_numpy(field.cpu().numpy(), data.resolution)).to(device) if needs_resize else field
            error = (readout.double()-u[:, step].double()).square().flatten(1).mean(1)
            power = u[:, step].double().square().flatten(1).mean(1)
            centered = u[:, step].double().clone()
            cpow = centered.square().flatten(1).mean(1)
            for key, value in (("mse", error), ("power", power), ("centered_power", cpow),
                               ("relative_l2", (error/power.clamp_min(1e-30)).sqrt())):
                batch[key].append(value.cpu().numpy())
            if probe_n:
                reference = model.encoder(low_u[:probe_n, step])
                rec = model.decoder(reference)
                dz = (z[:probe_n]-reference).double().square().flatten(1).mean(1).sqrt()
                decoded = (field[:probe_n]-rec).double().square().flatten(1).mean(1).sqrt()
                probe["rec_mse"].append((rec-low_u[:probe_n, step]).double().square().flatten(1).mean(1).cpu().numpy())
                probe["latent_rmse"].append(dz.cpu().numpy())
                probe["decoder_secant_gain"].append((decoded/dz.clamp_min(1e-30)).cpu().numpy())
        for key in values:
            values[key].append(np.stack(batch[key], axis=1))
        if probe_n:
            for key in probes:
                probes[key].append(np.stack(probe[key], axis=1))
        seen += len(u)
    arrays = {k: np.concatenate(v) for k, v in values.items()}
    arrays.update({k: np.concatenate(v) for k, v in probes.items() if v})
    mse, rel = arrays["mse"], arrays["relative_l2"]
    failed = ~(np.isfinite(mse).all(1) & np.isfinite(rel).all(1))
    split = getattr(data, "split", "val")
    result = {"split": split, "trajectories": count, "steps": mse.shape[1],
              "resolution": data.resolution, "training_resolution": training_resolution or data.resolution,
              "evaluation_protocol": "Downsample initial input, evolve at training resolution, upsample decoded readouts.",
              "nonfinite_trajectories": int(failed.sum()),
              "failed_indices": np.flatnonzero(failed).tolist(),
              "data_sha256": data.manifest["splits"][split]["sha256"],
              "probe_count": sum(len(v) for v in probes["rec_mse"]),
              "probe_note": "At training resolution, first fixed validation indices; directional decoder secant, not a Lipschitz bound."}
    if failed.any():
        # Do not silently discard failures or present a finite aggregate of survivors.
        result.update(first5_mse=None, blocks=[], mean_mse=None, mean_relative_l2=None)
        return result, arrays
    result.update(mean_mse=float(mse.mean()), first5_mse=float(mse[:, :5].mean()),
                  mean_relative_l2=float(rel.mean()),
                  final_relative_l2=float(rel[:, -1].mean()),
                  per_step_mse=mse.mean(0).tolist(),
                  per_step_relative_l2=rel.mean(0).tolist(),
                  per_step_p95_relative_l2=np.quantile(rel, .95, axis=0).tolist(),
                  p95_worst_step_relative_l2=float(np.quantile(rel.max(1), .95)),
                  trajectories_any_step_relative_l2_over_1=int((rel.max(1)>1).sum()),
                  centered_trajectory_nrmse=float(np.sqrt(mse.mean(1)/arrays["centered_power"].mean(1).clip(1e-30)).mean()),
                  blocks=[{"start_step": lo+1, "end_step": hi,
                           "mse": float(mse[:, lo:hi].mean()),
                           "relative_l2": float(rel[:, lo:hi].mean())}
                          for lo, hi in block_ranges(mse.shape[1])])
    if "rec_mse" in arrays:
        result["probe"] = {k: {"first": float(arrays[k][:, 0].mean()),
                                  "last": float(arrays[k][:, -1].mean())}
                           for k in probes}
    return result, arrays


def save_evaluation(output, result, arrays):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    # NPZ retains inf/nan when present; JSON uses an explicit failed result.
    np.savez_compressed(output/"per_trajectory.npz", **arrays)
    atomic_json(output/"metrics.json", result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--native", action="store_true")
    args = parser.parse_args()
    model, config, state, identity = initialize(args.run, args.device)
    resolution = config["data_generation"]["resolution"] if args.native else config["training_resolution"]
    data = checked_data(config, "val", resolution)
    result, arrays = evaluate(model, data, args.device, batch_size=min(8, config["training"]["batch_size"]),
                             limit=args.limit, training_resolution=config["training_resolution"])
    result.update(initial_checkpoint_sha256=identity, model=config["model"]["name"], source_sha256=source_hash(),
                  diagnostic_sha256=file_hash(Path(__file__)))
    save_evaluation(args.output, result, arrays)
    print(json.dumps({k: v for k, v in result.items() if not k.startswith("per_step")}, allow_nan=False))
