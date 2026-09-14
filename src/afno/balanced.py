"""A separately budgeted validation-only fine-tuning study, not a resumed paper run."""
import argparse
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from afno.data import atomic_json, file_hash
from afno.train import save_checkpoint, seed_all, source_hash
from .evaluation import checked_data, evaluate, initialize, save_evaluation
from .balanced_objectives import dual_score, alpha_at, rollout_loss


def trainer_hash():
    import hashlib
    digest = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        digest.update(path.name.encode()); digest.update(path.read_bytes())
    return digest.hexdigest()


def train(study, arm, run, output, references, device="cpu", stop_after=None):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    model, config, initial_state, identity = initialize(run, device)
    tc = study["training"]
    seed_all(tc["seed"])
    train_data = checked_data(config, "train", config["training_resolution"])
    val_data = checked_data(config, "val", config["training_resolution"])
    native_val = checked_data(config, "val", config["data_generation"]["resolution"])
    if tc["horizon"] != train_data.meta["steps"]:
        raise ValueError("This study trains complete trajectories from the initial input")
    if references["initial_checkpoint_sha256"] != identity:
        raise ValueError("Loss scales must belong to this exact initialization")
    if references["train_sha256"] != train_data.manifest["splits"]["train"]["sha256"]:
        raise ValueError("Loss scales were computed on different training data")
    comparator = references["fno_validation"]
    if comparator["data_sha256"] != val_data.manifest["splits"]["val"]["sha256"] or comparator["resolution"] != val_data.resolution:
        raise ValueError("FNO selection reference has different data or grid")
    if comparator["trajectories"] != len(val_data) or comparator["steps"] != tc["horizon"]:
        raise ValueError("FNO reference does not cover all trajectories and steps")
    batch_size = tc["batch_size"]
    optimizer = torch.optim.Adam(model.parameters(), lr=tc["learning_rate"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=tc["epochs"])
    provenance = {"kind": "dual_horizon_validation_only_finetune", "study": study, "arm": arm,
                  "initial_checkpoint_sha256": identity, "initial_checkpoint": str(Path(run)/"final.pt"),
                  "initial_source_sha256": source_hash(), "trainer_sha256": trainer_hash(),
                  "data_sha256": {s: train_data.manifest["splits"][s]["sha256"] for s in ("train", "val")},
                  "base_config": config, "torch": torch.__version__, "numpy": np.__version__,
                  "seed": tc["seed"], "references": references, "slurm_job_id": os.getenv("SLURM_JOB_ID"),
                  "tf32": False, "optimizer_reset": True, "test_evaluated": False}
    start, history, best_score, best_epoch = 0, [], None, 0
    last = output/"last.pt"
    if last.exists():
        state = torch.load(last, map_location=device, weights_only=False)
        old = dict(state["provenance"]); new = dict(provenance)
        old.pop("slurm_job_id", None); new.pop("slurm_job_id", None)
        if old != new:
            raise ValueError("Resume identity mismatch; use a new study, never extend a finished schedule")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start, history = state["epoch"], state["history"]
        best_score, best_epoch, initial_metrics = state["best_score"], state["best_epoch"], state["initial_metrics"]
        torch.set_rng_state(state["torch_rng"].cpu())
        if torch.device(device).type == "cuda":
            torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda_rng"]])
        np.random.set_state(state["numpy_rng"]); random.setstate(state["python_rng"])
    else:
        initial_metrics, arrays = evaluate(model, val_data, device, batch_size)
        if initial_metrics["nonfinite_trajectories"]:
            raise FloatingPointError("Initial validation is nonfinite; no eligible selection reference")
        save_evaluation(output/"initial_validation", initial_metrics, arrays)
        best_score = dual_score(initial_metrics, comparator)
        # Include the unmodified checkpoint as a selection candidate: never
        # silently select a degraded fine-tune simply because it trained last.
        save_checkpoint(output/"selected.pt", {"epoch": 0, "model": model.state_dict(),
                                              "provenance": provenance, "metrics": initial_metrics})
    atomic_json(output/"provenance.json", provenance)
    final_epoch = min(stop_after or tc["epochs"], tc["epochs"])
    for epoch in range(start+1, final_epoch+1):
        started = time.monotonic()
        horizon, alpha = tc["horizon"], alpha_at(epoch, arm)
        train_data.epoch = epoch-1
        generator = torch.Generator().manual_seed(tc["seed"]+epoch-1)
        loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, generator=generator, num_workers=0)
        model.train(); totals, count = {}, 0
        if torch.device(device).type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        for u, p, dt in loader:
            u, p, dt = u.to(device), p.to(device), dt.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss, terms = rollout_loss(model, u, p, dt, train_data.meta["lengths"],
                                       references["train_scales"], alpha, arm["mode"],
                                       tc["rec_weight"], tc["h1_weight"], tc["activation_checkpoint"])
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite loss, epoch {epoch}; no clipping or discarded trajectories")
            loss.backward()
            if not all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()):
                raise FloatingPointError(f"Missing/nonfinite gradient, epoch {epoch}")
            optimizer.step()
            count += len(u)
            for k, v in {"loss": loss.detach(), **terms}.items():
                totals[k] = totals.get(k, 0.) + v.item()*len(u)
        seconds = time.monotonic()-started
        lr = scheduler.get_last_lr()[0]; scheduler.step()
        row = {"epoch": epoch, "horizon": horizon, "early_weight": alpha, "learning_rate": lr,
               "train_seconds": seconds, "trajectories": count, "optimizer_updates": len(loader),
               "decoded_training_steps": count*horizon,
               "peak_cuda_bytes": torch.cuda.max_memory_allocated() if torch.device(device).type == "cuda" else None,
               **{k: v/count for k, v in totals.items()}}
        if epoch % tc["validate_every"] == 0 or epoch == tc["epochs"]:
            metrics, arrays = evaluate(model, val_data, device, batch_size)
            score = dual_score(metrics, comparator)
            row.update(val_mean_mse=metrics["mean_mse"], val_first5_mse=metrics["first5_mse"],
                       beats_fno_both=score is not None and score < 1, selection_score=score,
                       validation_nonfinite=metrics["nonfinite_trajectories"])
            val_output = output/"validation"/f"epoch_{epoch:03d}"
            val_output.mkdir(parents=True, exist_ok=True)
            save_checkpoint(val_output/"checkpoint.pt", {"epoch": epoch, "model": model.state_dict(),
                                                         "provenance": provenance, "metrics": metrics})
            metrics["checkpoint_sha256"] = file_hash(val_output/"checkpoint.pt")
            save_evaluation(val_output, metrics, arrays)
            if score is not None and score < best_score:
                best_score, best_epoch = score, epoch
                save_checkpoint(output/"selected.pt", {"epoch": epoch, "model": model.state_dict(),
                                                       "provenance": provenance, "metrics": metrics})
        history.append(row)
        atomic_json(output/"history.json", history)
        print(json.dumps(row, allow_nan=False), flush=True)
        state = {"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                 "scheduler": scheduler.state_dict(), "history": history, "provenance": provenance,
                 "best_score": best_score, "best_epoch": best_epoch, "initial_metrics": initial_metrics,
                 "torch_rng": torch.get_rng_state(),
                 "cuda_rng": torch.cuda.get_rng_state_all() if torch.device(device).type == "cuda" else [],
                 "numpy_rng": np.random.get_state(), "python_rng": random.getstate()}
        save_checkpoint(last, state)
        if epoch == tc["epochs"]:
            save_checkpoint(output/"final.pt", state)
    if final_epoch == tc["epochs"]:
        # Final and selected are both disclosed. Selection already happened
        # at training resolution; the native-grid validation cannot change it.
        for label in ("final", "selected"):
            checkpoint_path = output/f"{label}.pt"
            state = torch.load(checkpoint_path, map_location=device, weights_only=False)
            model.load_state_dict(state["model"])
            metrics, arrays = evaluate(model, native_val, device, min(8, batch_size),
                                       training_resolution=config["training_resolution"])
            metrics.update(epoch=state["epoch"], checkpoint_sha256=file_hash(checkpoint_path))
            save_evaluation(output/f"{label}_native_validation", metrics, arrays)
    atomic_json(output/"status.json", {"epochs_completed": final_epoch, "planned_epochs": tc["epochs"],
                                      "selected_epoch": best_epoch, "selected_training_grid_score": best_score,
                                      "beats_fno_both_on_selection_grid": best_score < 1,
                                      "complete": final_epoch == tc["epochs"],
                                      "test_evaluated": False})


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--study", required=True)
    p.add_argument("--arm", required=True)
    p.add_argument("--run", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--references", required=True)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    a = p.parse_args()
    study = json.loads(Path(a.study).read_text())
    arm = next(arm for arm in study["arms"] if arm["name"] == a.arm)
    train(study, arm, a.run, a.output, json.loads(Path(a.references).read_text()), a.device)
