"""Memory-mapped trajectories with explicit, independent train/val/test splits."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from filelock import FileLock
import hashlib
import json
import multiprocessing
from pathlib import Path
import time

import numpy as np
from scipy.signal import resample
import torch
from torch.utils.data import Dataset

from .solvers import initial_conditions, solve


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8*1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, data):
    path = Path(path)
    tmp = path.with_suffix(path.suffix+".tmp")
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False)+"\n")
    tmp.replace(path)


def resize_numpy(array, shape):
    for axis, n in zip(range(array.ndim-len(shape), array.ndim), shape):
        if array.shape[axis] != n:
            array = resample(array, n, axis=axis)
    return np.asarray(array, dtype=np.float32)


def _solve_batch(task):
    config, seeds, lo, hi = task
    initial = np.concatenate([initial_conditions(config, seeds[i], 1) for i in range(lo, hi)])
    return lo, hi, solve(config, initial).astype(np.float32)


def generate(config, output, split_names=None, batch_size=8, workers=1):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with FileLock(str(output/".generation.lock"), timeout=0):
        return _generate(config, output, split_names, batch_size, workers)


def _generate(config, output, split_names, batch_size, workers):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output/"manifest.json"
    existing = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    if existing and existing["config"] != config:
        raise ValueError("Existing dataset has a different configuration; choose a new directory")
    manifest = existing or {"schema_version": 1, "provenance": "independent_solver_assumptions",
        "generator_sha256": {name: file_hash(Path(__file__).parent/name)
                              for name in ("data.py", "solvers.py")},
        "config": config, "splits": {}}
    for split in split_names or config["splits"]:
        count = config["splits"][split]
        seed = config["split_seeds"][split]
        path = output/f"{split}.npy"
        if split in manifest["splits"]:
            if not path.exists() or file_hash(path) != manifest["splits"][split]["sha256"]:
                raise ValueError(f"Existing {split} data hash mismatch")
            print(f"Verified existing {split}: {path}", flush=True)
            continue
        tmp = output/f"{split}.partial.npy"
        shape = (count, config["steps"]+1, config["channels"], *config["resolution"])
        array = np.lib.format.open_memmap(tmp, mode="w+", dtype="float32", shape=shape)
        start = time.monotonic()
        # Per-trajectory seeds preserve data identity across generation batch sizes.
        seeds = np.random.SeedSequence(seed).spawn(count)
        tasks = [(config, seeds, lo, min(lo+batch_size, count)) for lo in range(0, count, batch_size)]
        pool = ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) if workers > 1 else None
        try:
            chunks = pool.map(_solve_batch, tasks) if pool else map(_solve_batch, tasks)
            for lo, hi, values in chunks:
                array[lo:hi] = values
                array.flush()
                print(json.dumps({"split": split, "generated": hi, "total": count,
                                  "elapsed_seconds": round(time.monotonic()-start, 2)}), flush=True)
        finally:
            if pool:
                pool.shutdown(wait=True, cancel_futures=True)
        del array
        tmp.replace(path)
        manifest["splits"][split] = {"path": path.name, "shape": shape,
            "dtype": "float32", "seed": seed, "sha256": file_hash(path),
            "generation_seconds": time.monotonic()-start}
        atomic_json(manifest_path, manifest)
    return manifest


class TrajectoryDataset(Dataset):
    def __init__(self, root, split, resolution, horizon=None, seed=0):
        root = Path(root)
        self.manifest = json.loads((root/"manifest.json").read_text())
        self.meta = self.manifest["config"]
        entry = self.manifest["splits"][split]
        self.array = np.load(root/entry["path"], mmap_mode="r")
        if tuple(self.array.shape) != tuple(entry["shape"]):
            raise ValueError("Manifest and trajectory shape disagree")
        self.resolution, self.horizon = resolution, horizon
        if horizon is not None and not 1 <= horizon < self.array.shape[1]:
            raise ValueError("Invalid training horizon")
        self.seed, self.epoch = seed, 0
        self.params = torch.tensor([self.meta["params"][k] for k in self.meta["conditioning"]], dtype=torch.float32)
        self.dt = torch.tensor(self.meta["observation_dt"], dtype=torch.float32)

    def __len__(self):
        return len(self.array)

    def __getitem__(self, i):
        if self.horizon is not None:
            rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch, int(i)]))
            start = int(rng.integers(0, self.array.shape[1]-self.horizon))
            u = self.array[i, start:start+self.horizon+1]
        else:
            u = self.array[i]
        u = resize_numpy(np.array(u, copy=True), self.resolution)
        return torch.from_numpy(u), self.params.clone(), self.dt.clone()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--splits", nargs="+", choices=["train", "val", "test"])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    generate(config.get("data_generation", config), args.output, args.splits, args.batch_size, args.workers)


if __name__ == "__main__":
    main()
