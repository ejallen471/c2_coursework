#!/bin/bash
#SBATCH --job-name=st_block_sweep
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=slurm-%x-%j.out

# Submit on CSD3 with your account and partition, for example:
# sbatch --account=<ACCOUNT> --partition=<PARTITION> scripts/slurm_scripts/single_thread_block_size_sweep.slurm

set -euo pipefail

ROOT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${ROOT_DIR}"

OPTIMISATION_METHOD="${1:-cache_blocked_1}"
MATRIX_SIZE="${2:-1000}"
REPEATS="${3:-5}"

shift $(( $# >= 3 ? 3 : $# ))

if [ "$#" -eq 0 ]; then
    BLOCK_SIZES=(16 24 32 48 64)
else
    BLOCK_SIZES=("$@")
fi

if type module >/dev/null 2>&1; then
    module purge
    module load cmake
    module load gcc
fi

# Load your Python/Conda environment here if CSD3 does not already expose `conda`.

bash scripts/sh_scripts/run_block_size_sweep.sh \
    "${OPTIMISATION_METHOD}" \
    "${MATRIX_SIZE}" \
    "${REPEATS}" \
    "${BLOCK_SIZES[@]}"
