"""Portable, resumable FNO / AFNO training workflow for the two examples."""
from copy import deepcopy
import json
from pathlib import Path

from filelock import FileLock

from . import balanced, warmup
from .calibration import calibrate
from .data import atomic_json, file_hash, generate
from .evaluation import checked_data, evaluate, initialize, save_evaluation
from .train import source_hash, train as base_train

ASSETS = Path(__file__).resolve().parent / "assets"


def preset(name):
    """Return an independent copy of a bundled JSON recipe."""
    if name not in ("burgers", "ks", "smoke_burgers", "warmup", "balanced"):
        raise ValueError(f"Unknown preset: {name}")
    return json.loads((ASSETS / "configs" / f"{name}.json").read_text())


def recipes(equation, smoke=False):
    config, first, second = preset(equation), preset("warmup"), preset("balanced")
    if smoke:
        config = preset("smoke_burgers")
        config["name"] = f"smoke_{equation}"
        if equation == "ks":
            reference = preset("ks")
            for key in ("equation", "lengths", "observation_dt", "internal_dt", "params", "conditioning"):
                config["data_generation"][key] = reference["data_generation"][key]
            config["model"]["physical_params"] = 0
        # Exercise both reference-grid readout interpolation and all training stages.
        config["data_generation"]["resolution"] = [64]
        first["training"].update(epochs=4, batch_size=4, validate_every=2)
        first["arms"][0]["stages"] = [
            {"through_epoch": 2, "horizon": 5}, {"through_epoch": 4, "horizon": 8}]
        second["training"].update(epochs=4, horizon=8, batch_size=4, validate_every=2)
        second["arms"][0]["stages"] = [{"through_epoch": 4, "early_weight": .5}]
    return config, first, second


def run(equation, output, device="cpu", smoke=False, resume=False, workers=1,
        config=None, warmup_config=None, balanced_config=None):
    """Generate train/val, train FNO, then AFNO base -> warmup -> balanced.

    Resume retains the original optimizer budgets and rejects changed code,
    recipes or data. Completed stages are verified and reused.
    """
    if equation not in ("burgers", "ks"):
        raise ValueError("Choose burgers or ks")
    root = Path(output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    with FileLock(str(root / ".workflow.lock"), timeout=0):
        return _run(equation, root, device, smoke, resume, workers,
                    config, warmup_config, balanced_config)


def _run(equation, root, device, smoke, resume, workers, config, first, second):
    defaults = recipes(equation, smoke)
    config, first, second = [deepcopy(x if x is not None else y)
                             for x, y in zip((config, first, second), defaults)]
    if config["data_generation"]["equation"] != equation:
        raise ValueError("Configuration and requested equation differ")
    if len(first["arms"]) != 1 or len(second["arms"]) != 1:
        raise ValueError("This workflow runs one warmup and one balanced recipe")
    if config["model"]["name"] != "afno":
        raise ValueError("The workflow configuration must specify AFNO; FNO is derived from it")
    config["data_path"] = str(root / "data")
    definition = {"equation": equation, "smoke": smoke, "config": config,
                  "warmup": first, "balanced": second, "source_sha256": source_hash()}
    plan = root / "workflow.json"
    if plan.exists():
        if not resume:
            raise FileExistsError("Existing workflow: use --resume or a new output directory")
        if json.loads(plan.read_text()) != definition:
            raise ValueError("Workflow changed; resume requires the original code and recipes")
    elif resume:
        raise FileNotFoundError("--resume requires an existing workflow.json")
    else:
        atomic_json(plan, definition)
    generate(config["data_generation"], config["data_path"], ["train", "val"], workers=workers)
    # Every resume verifies data, including when every model stage is complete.
    for split in ("train", "val"):
        checked_data(config, split, config["training_resolution"])

    def stage_complete(name, expected):
        marker = root / name / "complete.json"
        if not marker.exists():
            return False
        saved = json.loads(marker.read_text())
        if saved["identity"] != expected:
            raise ValueError(f"{name} stage identity changed")
        for filename, digest in saved["artifacts"].items():
            if file_hash(root / name / filename) != digest:
                raise ValueError(f"{name}/{filename} changed after completion")
        print(f"Verified completed stage: {name}", flush=True)
        return True

    def complete(name, identity, artifacts):
        atomic_json(root / name / "complete.json", {
            "identity": identity,
            "artifacts": {f: file_hash(root / name / f) for f in artifacts}})

    base_paths = {}
    for name in ("fno", "afno"):
        stage = f"{name}_base"
        cfg = deepcopy(config)
        cfg["model"]["name"] = name
        identity = {"config": cfg, "source_sha256": source_hash()}
        if not stage_complete(stage, identity):
            base_train(cfg, root / stage, device, resume=(root / stage / "last.pt").exists())
            complete(stage, identity, ["final.pt", "status.json"])
        base_paths[name] = root / stage

    fno, _, _, fno_sha = initialize(base_paths["fno"], device)
    validation = checked_data(config, "val", config["training_resolution"])
    reference, arrays = evaluate(fno, validation, device, min(8, config["training"]["batch_size"]), probe_count=0)
    reference.update(model="fno", checkpoint_sha256=fno_sha)
    if reference["nonfinite_trajectories"] or min(reference["first5_mse"], reference["mean_mse"]) <= 0:
        raise FloatingPointError("FNO reference must have finite, positive errors")
    save_evaluation(root / "fno_validation", reference, arrays)
    native = checked_data(config, "val", config["data_generation"]["resolution"])
    metrics, arrays = evaluate(fno, native, device, min(8, config["training"]["batch_size"]),
                               probe_count=0, training_resolution=config["training_resolution"])
    save_evaluation(root / "fno_native_validation", metrics, arrays)
    del fno

    identity = {"initial_sha256": file_hash(base_paths["afno"] / "final.pt"), "study": first}
    if not stage_complete("afno_warmup", identity):
        warmup.train(first, first["arms"][0], base_paths["afno"], root / "afno_warmup", device)
        complete("afno_warmup", identity, ["final.pt", "selected.pt", "status.json"])

    identity = {"initial_sha256": file_hash(root / "afno_warmup/final.pt"),
                "study": second, "fno_reference": reference}
    if not stage_complete("calibration", identity):
        calibrate(root / "afno_warmup", root / "calibration", reference, second, device)
        complete("calibration", identity, ["references.json", "gate.json"])
    references = json.loads((root / "calibration/references.json").read_text())
    identity = {"references": references, "study": second}
    if not stage_complete("afno", identity):
        balanced.train(second, second["arms"][0], root / "afno_warmup", root / "afno", references, device)
        complete("afno", identity, ["final.pt", "selected.pt", "status.json"])
    summary = {"equation": equation, "smoke": smoke, "complete": True,
               "test_evaluated": False, "validation": {
                   "FNO": json.loads((root / "fno_native_validation/metrics.json").read_text()),
                   "AFNO": json.loads((root / "afno/final_native_validation/metrics.json").read_text())}}
    atomic_json(root / "summary.json", summary)
    print(f"Completed {equation}; results: {root / 'summary.json'}", flush=True)
    return summary
