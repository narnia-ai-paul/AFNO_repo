# Reproducibility

## Three ways to use the project

1. **Saved figures:** `python -m afno figures` reads the bundled saved predictions and errors. It reproduces the plotted values without model inference.
2. **Pretrained inference:** `python -m afno demo` loads the four bundled model state dictionaries and recomputes trajectory-zero predictions. It reports differences from saved fields. Small floating-point differences may arise from hardware, library versions or inference batch size.
3. **Retraining:** `python -m afno train` generates the data and runs the complete recipe. The port preserves the model and objective mathematics, but the full training budgets have not been repeated in this standalone project. Identical numerical outcomes across environments are not guaranteed.

The full reference environment is recorded in [requirements-tested.txt](../requirements-tested.txt). It describes the environment used to validate this release, not a requirement that every future installation match it exactly. For closest numerical comparisons, use Python 3.11 and those versions in a dedicated environment. No original cluster path or Slurm allocation is required by the code.

## Assets

`src/afno/assets/manifest.json` contains relative filenames, SHA-256 hashes, model configurations, source/checkpoint identities from the original experiments, data-generation metadata, population metrics, and the figure caption. `python -m afno verify` checks all bundled data, configuration and weight hashes. Release-level checksums are also listed in `SHA256SUMS`.

Each `examples/<equation>_sample.npz` contains:

| Key | Shape | Meaning |
| --- | --- | --- |
| `initial` | (1, 1, 1024) | Ground-truth initial field |
| `target` | (1, 100, 1, 1024) | Future ground-truth fields |
| `fno_prediction`, `afno_prediction` | (1, 100, 1, 1024) | Saved native-grid readouts |
| `x`, `time` | (1024,), (100,) | Spatial coordinates and future times |
| `indices` | (1,) | Validation trajectory index, always 0 |

Each `<equation>_population.npz` contains `fno_mse` and `afno_mse`, each shaped `(100, 100)`, plus trajectory indices and step numbers. Those arrays retain every trajectory and prediction step. Full-population predictions are omitted to keep the shared project small; the population curves can still be regenerated exactly from the saved MSE arrays. Regenerating all predictions requires generating validation data and evaluating a model on it.

Each distributed `.pt` file contains `state_dict`, `model_config`, `training_resolution`, `data_generation`, and epoch metadata. It is loaded with `weights_only=True`. These are inference checkpoints, not optimizer-resume checkpoints. Newly trained runs separately save model, optimizer, scheduler and random-number-generator states for exact interrupted continuation within the same environment. Load those training checkpoints only from trusted runs.

## Training outputs

Within the chosen output directory:

```text
workflow.json                 Source hash and complete workflow definition
data/                         Generated train.npy, val.npy and manifest
fno_base/final.pt              FNO checkpoint
afno_base/final.pt             AFNO base checkpoint
afno_warmup/final.pt           AFNO after the increasing-horizon stage
calibration/references.json   Frozen training-only normalization errors
afno/final.pt                 AFNO after balanced training; figure protocol
afno/selected.pt              Separate validation-selected candidate
fno_native_validation/        Full FNO validation metrics and arrays
afno/final_native_validation/ Full final-AFNO validation metrics and arrays
summary.json                  Side-by-side final validation results
```

Logs, checkpoint histories and per-trajectory errors remain in each run. Checkpoint and data checks prevent accidental reuse after source or recipe changes. Resume the same output location with `--resume`; moving an in-progress run or changing its source/configuration is not an automatic migration. Independent copies of the distributable project can start new runs anywhere.

Evaluate a newly trained final checkpoint:

```bash
python -m afno evaluate --run runs/burgers/afno --native \
  --output outputs/burgers_evaluation --device cuda
```

The evaluation command accepts the newly trained run format. For bundled inference checkpoints use `demo`. This workflow evaluates validation data only; the full recipe metadata preserves the independent test split's count and seed for provenance, but the standard commands neither generate nor evaluate that split.

## Sharing

The source tree is a standalone local Git repository. The accompanying ZIP contains the project files, examples, pretrained weights, figures and tests; it excludes `.git`, virtual environments, generated datasets and run directories. Extract it, follow the README, and use it without access to the original research workspace. No remote repository or external publication is configured.
