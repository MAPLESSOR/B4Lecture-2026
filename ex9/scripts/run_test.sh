#!/usr/bin/env bash
#SBATCH --job-name=forB4Lecture    # Job name
#SBATCH --mem=32G                   # Memory total in GB (for all cores)
#SBATCH --time=24:00:00             # Time limit hrs:min:sec
#SBATCH --output=batchlog/%j/job_output_%j.log  # output.log
#SBATCH --error=batchlog/%j/job_error_%j.log    # error.log
#SBATCH --gres=gpu:1                # Number of GPUs to use

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

"${PYTHON}" -m pytest