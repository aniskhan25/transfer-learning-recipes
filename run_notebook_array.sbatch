#!/bin/bash
#SBATCH --account=project_2014553
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --gres=gpu:v100:1
#SBATCH --job-name=nb
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --array=0-7

set -euo pipefail

source /scratch/project_2014553/anisrahm/venvs/marimo/bin/activate
export PYTHONPATH=/scratch/project_2014553/anisrahm/transfer-learning-recipes/src:${PYTHONPATH:-}

ROOT="/scratch/project_2014553/anisrahm/transfer-learning-recipes"
NOTEBOOKS=(
  "01_baseline_from_scratch"
  "02_transfer_core_loop"
  "03_safe_transfer_staged_adaptation"
  "04_naive_transfer_and_forgetting"
  "05_related_vs_unrelated_tasks"
  "06_stability_mismatch_proxy"
  "07_modern_methods_and_label_budget"
  "08_design_pattern_and_checklist"
)

if [ -n "${NB_NAME:-}" ]; then
  NB="$NB_NAME"
  if [ "${SLURM_ARRAY_TASK_ID:-0}" -ne 0 ]; then
    exit 0
  fi
else
  NB="${NOTEBOOKS[$SLURM_ARRAY_TASK_ID]}"
fi
NB_IPYNB="$ROOT/notebooks/${NB}.ipynb"
NB_MO="$ROOT/notebooks/${NB}.mo.py"
OUT_HTML="$ROOT/outputs/html/${NB}.html"

mkdir -p "$ROOT/outputs/html" "$ROOT/logs"

marimo convert "$NB_IPYNB" -o "$NB_MO"
marimo export html "$NB_MO" -o "$OUT_HTML"