#!/usr/bin/env bash
# Example only: adjust resource directives to your site's scheduler configuration.
#SBATCH --job-name=afno-example
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=afno-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit this script from the project root}"
source .venv/bin/activate
export OMP_NUM_THREADS=4
equation="${1:-burgers}"
python -m afno train --equation "$equation" --output "runs/$equation" --device cuda "${@:2}"
