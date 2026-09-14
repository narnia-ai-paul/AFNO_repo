"""Fixed-budget training; no test data access; atomic, resumable checkpoints."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import TrajectoryDataset, atomic_json, file_hash
from .losses import training_loss
from .models import AFNO, build_model


def source_hash():
    digest = hashlib.sha256()
    for p in sorted(Path(__file__).parent.glob("*.py")):
        digest.update(p.name.encode()); digest.update(p.read_bytes())
    return digest.hexdigest()


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def save_checkpoint(path, state):
    path = Path(path); tmp = path.with_suffix(".tmp")
    torch.save(state, tmp); tmp.replace(path)


def provenance(config, dataset):
    return {"config": config, "source_sha256": source_hash(),
            "data_manifest": dataset.manifest, "python": platform.python_version(),
            "torch": torch.__version__, "numpy": np.__version__,
            "hostname": platform.node(), "slurm_job_id": os.getenv("SLURM_JOB_ID"),
            "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
            "numerical_policy": {"cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
                "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                "cudnn_benchmark": torch.backends.cudnn.benchmark}}


@torch.no_grad()
def validate(model, dataset, device, batch_size, limit=32):
    model.eval()
    indices = range(min(limit, len(dataset)))
    loader = DataLoader(torch.utils.data.Subset(dataset, indices), batch_size=batch_size)
    mse, total = 0.0, 0
    for u, p, dt in loader:
        u, p, dt = u.to(device), p.to(device), dt.to(device)
        pred = model.rollout(u[:, 0], p, dt, u.shape[1]-1)
        value = (pred-u[:, 1:]).square().flatten(1).mean(1)
        if not torch.isfinite(value).all():
            raise FloatingPointError("Nonfinite validation rollout")
        mse += value.sum().item(); total += len(u)
    return mse/total


def train(config, run_dir, device="cpu", resume=False, stop_after=None):
    run_dir = Path(run_dir); run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = run_dir/"last.pt"
    if checkpoint_path.exists() and not resume:
        raise FileExistsError("Existing checkpoint: use --resume or a new run directory")
    if resume and not checkpoint_path.exists():
        raise FileNotFoundError("--resume requires last.pt")
    tc = config["training"]
    seed_all(tc["seed"])
    torch.set_num_threads(int(os.getenv("OMP_NUM_THREADS", "2")))
    device = torch.device(device)
    train_data = TrajectoryDataset(config["data_path"], "train", config["training_resolution"],
                                   tc["horizon"], seed=tc["seed"])
    val_data = TrajectoryDataset(config["data_path"], "val", config["training_resolution"],
                                 tc["horizon"], seed=tc["seed"]+10000)
    if train_data.meta != config["data_generation"]:
        raise ValueError("Training data configuration differs from the declared experiment")
    # Verify only train/val. This does not open the test trajectory file.
    for split in ("train", "val"):
        entry = train_data.manifest["splits"][split]
        if file_hash(Path(config["data_path"])/entry["path"]) != entry["sha256"]:
            raise ValueError(f"{split} hash mismatch")
    model = build_model(config["model"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=tc["learning_rate"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=tc["epochs"])
    meta = provenance(config, train_data)
    meta["parameter_count_real"] = sum(p.numel()*(2 if p.is_complex() else 1) for p in model.parameters())
    start_epoch, history = 0, []
    if resume:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if checkpoint["provenance"]["config"] != config:
            raise ValueError("Resume config mismatch; never extend a finished cosine schedule")
        if checkpoint["provenance"]["source_sha256"] != meta["source_sha256"]:
            raise ValueError("Source changed since checkpoint; use a new run for changed code")
        old = checkpoint["provenance"]["data_manifest"]["splits"]
        if any(old[s]["sha256"] != train_data.manifest["splits"][s]["sha256"] for s in ("train", "val")):
            raise ValueError("Resume data mismatch")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = checkpoint["epoch"]
        history = checkpoint["history"]
        torch.set_rng_state(checkpoint["torch_rng"].cpu())
        if device.type == "cuda":
            torch.cuda.set_rng_state_all([s.cpu() for s in checkpoint["cuda_rng"]])
        np.random.set_state(checkpoint["numpy_rng"])
        random.setstate(checkpoint["python_rng"])
        meta = checkpoint["provenance"]
    atomic_json(run_dir/"provenance.json", meta)
    atomic_json(run_dir/"config.json", config)
    warmup_epochs = int(tc["epochs"]*tc.get("reconstruction_fraction", 0.3))
    final_epoch = min(tc["epochs"], stop_after) if stop_after else tc["epochs"]
    for epoch in range(start_epoch, final_epoch):
        started = time.monotonic()
        model.train(); train_data.epoch = epoch
        gen = torch.Generator().manual_seed(tc["seed"]+epoch)
        loader = DataLoader(train_data, batch_size=tc["batch_size"], shuffle=True,
                            generator=gen, num_workers=0, pin_memory=device.type == "cuda")
        weights = ([1., 0., 0.] if epoch < warmup_epochs else tc["stage2_weights"]) if isinstance(model, AFNO) else [0., 0., 1.]
        totals, count = {}, 0
        for u, p, dt in loader:
            u, p, dt = u.to(device), p.to(device), dt.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss, terms = training_loss(model, u, p, dt, train_data.meta["lengths"], weights,
                                        tc["h1_weight"], tc.get("pushforward_steps", 0))
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite loss at epoch {epoch+1}")
            loss.backward()
            if not all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
                raise FloatingPointError(f"Nonfinite gradient at epoch {epoch+1}")
            optimizer.step()
            for key, value in {"loss": loss.detach(), **terms}.items():
                totals[key] = totals.get(key, 0.)+value.item()*len(u)
            count += len(u)
        used_lr = scheduler.get_last_lr()[0]
        scheduler.step()
        phase = ("reconstruction" if epoch < warmup_epochs else "dynamics") if isinstance(model, AFNO) else "supervised_rollout"
        row = {"epoch": epoch+1, "phase": phase,
               "learning_rate": used_lr, "weights": weights,
               **{k: v/count for k, v in totals.items()},
               "train_seconds": time.monotonic()-started}
        if (epoch+1) % tc.get("validate_every", 25) == 0 or epoch+1 == final_epoch:
            row["val_rollout_mse"] = validate(model, val_data, device, tc["batch_size"])
        history.append(row)
        print(json.dumps(row, allow_nan=False), flush=True)
        atomic_json(run_dir/"history.json", history)
        if (epoch+1) % tc.get("checkpoint_every", 25) == 0 or epoch+1 == final_epoch:
            state = {"schema_version": 1, "epoch": epoch+1, "model": model.state_dict(),
                "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                "provenance": meta, "history": history, "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
                "numpy_rng": np.random.get_state(), "python_rng": random.getstate()}
            save_checkpoint(checkpoint_path, state)
            if epoch+1 == tc["epochs"]:
                save_checkpoint(run_dir/"final.pt", state)
    atomic_json(run_dir/"status.json", {"completed_epochs": max(start_epoch, final_epoch),
        "planned_epochs": tc["epochs"], "complete": max(start_epoch, final_epoch) == tc["epochs"],
        "test_evaluated": False})
    return history


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after", type=int, help="Interrupt the original budget for testing resume")
    args = parser.parse_args()
    train(json.loads(Path(args.config).read_text()), args.run_dir, args.device, args.resume, args.stop_after)


if __name__ == "__main__":
    main()
