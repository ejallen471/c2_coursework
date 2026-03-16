#!/bin/bash
#SBATCH --job-name=omp_fixed_size
#SBATCH --account=MPHIL-DIS-SL2-CPU
#SBATCH --partition=sapphire
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --output=logs/fixed_size_comparision/slurm-%x-%j.out
#SBATCH --error=logs/fixed_size_comparision/slurm-%x-%j.err

set -euo pipefail

ROOT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${ROOT_DIR}"

MATRIX_SIZE="${1:-2000}"
REPEATS="${2:-1}"
OUTPUT_CSV="${3:-results/raw/openmp_fixed_size_comparison_n${MATRIX_SIZE}.csv}"
shift $(( $# >= 3 ? 3 : $# ))
METHODS=("$@")
EXTRA_ARGS=()

if [ "${#METHODS[@]}" -gt 0 ]; then
    EXTRA_ARGS=(--methods "${METHODS[@]}")
fi

module purge
module load cmake
module load gcc
module load python

source ~/venvs/c2_coursework/bin/activate

export OPENMP_THREADS="${OPENMP_THREADS:-${SLURM_CPUS_PER_TASK}}"

bash scripts/sh_scripts/run_openmp_fixed_size_comparison.sh \
    "${MATRIX_SIZE}" \
    "${REPEATS}" \
    "${OUTPUT_CSV}" \
    "${EXTRA_ARGS[@]}"
