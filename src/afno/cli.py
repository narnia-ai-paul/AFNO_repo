"""Command-line entry points. Run `afno --help` after installation."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="AFNO: Burgers and Kuramoto–Sivashinsky examples")
    sub = parser.add_subparsers(dest="command", required=True)
    demo = sub.add_parser("demo", help="Run pretrained 100-step predictions on bundled examples")
    demo.add_argument("--equation", choices=["burgers", "ks", "both"], default="both")
    demo.add_argument("--output", default="outputs/demo")
    demo.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    demo.add_argument("--no-figures", action="store_true")
    figures = sub.add_parser("figures", help="Recreate academic figures from saved validation predictions")
    figures.add_argument("--output", default="outputs/figures")
    verify = sub.add_parser("verify", help="Verify all bundled example and weight checksums")
    for command, help_text in [("train", "Run the full FNO and AFNO training recipe"),
                                ("smoke", "Exercise the full pipeline with tiny CPU-scale models")]:
        p = sub.add_parser(command, help=help_text)
        p.add_argument("--equation", choices=["burgers", "ks"], required=True)
        p.add_argument("--output", required=True)
        p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
        p.add_argument("--resume", action="store_true")
        p.add_argument("--workers", type=int, default=1, help="Data-generation processes")
        if command == "train":
            for option in ("config", "warmup-config", "balanced-config"):
                p.add_argument("--" + option, help="Optional custom JSON recipe")
    p = sub.add_parser("evaluate", help="Evaluate a locally trained final checkpoint on all validation trajectories")
    p.add_argument("--run", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--native", action="store_true", help="Interpolate readouts to the reference grid")
    args = parser.parse_args()
    if args.command == "demo":
        from .demo import run
        run(args.equation, args.output, args.device, not args.no_figures)
    elif args.command == "figures":
        from .figures import render
        render(args.output)
    elif args.command == "verify":
        from .demo import verify_assets
        print(json.dumps(verify_assets(), indent=2))
    elif args.command in ("train", "smoke"):
        from .workflow import run
        extra = {key: json.loads(Path(getattr(args, key)).read_text())
                 for key in ("config", "warmup_config", "balanced_config") if getattr(args, key, None)}
        run(args.equation, args.output, args.device, args.command == "smoke", args.resume, args.workers, **extra)
    elif args.command == "evaluate":
        from .evaluation import checked_data, evaluate, initialize, save_evaluation
        model, config, _, identity = initialize(args.run, args.device)
        resolution = config["data_generation"]["resolution"] if args.native else config["training_resolution"]
        data = checked_data(config, "val", resolution)
        metrics, arrays = evaluate(model, data, args.device, training_resolution=config["training_resolution"])
        metrics["checkpoint_sha256"] = identity
        save_evaluation(args.output, metrics, arrays)
        print(f"Saved validation metrics to {args.output}")
