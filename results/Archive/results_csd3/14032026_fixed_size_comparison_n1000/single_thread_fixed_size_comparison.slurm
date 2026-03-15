#!/bin/bash
#SBATCH --job-name=st_fixed_size
#SBATCH --account=MPHIL-DIS-SL2-CPU
#SBATCH --partition=sapphire
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00
#SBATCH --output=logs/fixed_size_comparison/slurm-%x-%j.out
#SBATCH --error=logs/fixed_size_comparison/slurm-%x-%j.err

set -eo pipefail

ROOT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${ROOT_DIR}"

MATRIX_SIZE="${1:-1000}"
REPEATS="${2:-1}"

if type module >/dev/null 2>&1; then
    module purge
    module load cmake
    module load gcc
fi

bash scripts/sh_scripts/build.sh

./build/run/run_cholesky \
    fixed-size \
    "${MATRIX_SIZE}" \
    "${REPEATS}" \
    "results/raw/fixed_size_comparison_n${MATRIX_SIZE}_raw.csv"
