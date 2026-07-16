#!/usr/bin/env bash
#SBATCH --job-name=forB4Lecture    # Job name
#SBATCH --mem=32G                   # Memory total in GB (for all cores)
#SBATCH --time=24:00:00             # Time limit hrs:min:sec
#SBATCH --output=batchlog/%j/job_output_%j.log  # output.log
#SBATCH --error=batchlog/%j/job_error_%j.log    # error.log
#SBATCH --gres=gpu:1                # Number of GPUs to use
set -euo pipefail

# ディレクトリの位置を変えるよう修正
ROOT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${ROOT_DIR}"

export PYTHONPATH="${ROOT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"

PYTHON=${PYTHON:-python}
DEVICE=${DEVICE:-auto}
TRAIN_OUTPUT_DIR=${TRAIN_OUTPUT_DIR:-runs/toy_realnvp}
SAMPLE_OUTPUT_DIR=${SAMPLE_OUTPUT_DIR:-outputs/toy_realnvp}
DENSITY_GRID_SIZE=${DENSITY_GRID_SIZE:-180}
BASE_GRID_SIZE=${BASE_GRID_SIZE:-17}

train_args=()
sample_args=()

if [[ -n "${NUM_STEPS:-}" ]]; then
  train_args+=(--num-steps "${NUM_STEPS}")
fi
if [[ -n "${BATCH_SIZE:-}" ]]; then
  train_args+=(--batch-size "${BATCH_SIZE}")
fi
if [[ -n "${NUM_SAMPLES:-}" ]]; then
  sample_args+=(--num-samples "${NUM_SAMPLES}")
fi
if [[ -n "${SEED:-}" ]]; then
  sample_args+=(--seed "${SEED}")
fi

"${PYTHON}" scripts/train_toy.py \
  --device "${DEVICE}" \
  --output-dir "${TRAIN_OUTPUT_DIR}" \
  "${train_args[@]}"

"${PYTHON}" scripts/sample_toy.py \
  --device "${DEVICE}" \
  --checkpoint "${TRAIN_OUTPUT_DIR}/checkpoint.pt" \
  --output-dir "${SAMPLE_OUTPUT_DIR}" \
  "${sample_args[@]}"

"${PYTHON}" scripts/plot_toy.py \
  --device "${DEVICE}" \
  --train-output-dir "${TRAIN_OUTPUT_DIR}" \
  --sample-output-dir "${SAMPLE_OUTPUT_DIR}" \
  --output-dir "${SAMPLE_OUTPUT_DIR}" \
  --density-grid-size "${DENSITY_GRID_SIZE}" \
  --base-grid-size "${BASE_GRID_SIZE}"
