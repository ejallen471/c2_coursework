#!/bin/bash
#SBATCH --job-name=omp_block_sweep
#SBATCH --account=MPHIL-DIS-SL2-CPU
#SBATCH --partition=sapphire
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

ROOT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${ROOT_DIR}"

OPTIMISATION_METHOD="${1:-openmp3}"
MATRIX_SIZE="${2:-2000}"
REPEATS="${3:-5}"

shift $(( $# >= 3 ? 3 : $# ))

if [ "$#" -eq 0 ]; then
    BLOCK_SIZES=(8 12 16 24 32 40 48 64 80 96 128)
else
    BLOCK_SIZES=("$@")
fi

module purge
module load cmake
module load gcc
module load python

source ~/venvs/c2_coursework/bin/activate

export OPENMP_THREADS="${OPENMP_THREADS:-${SLURM_CPUS_PER_TASK}}"

bash scripts/sh_scripts/run_openmp_block_size_sweep.sh \
    "${OPTIMISATION_METHOD}" \
    "${MATRIX_SIZE}" \
    "${REPEATS}" \
    "${BLOCK_SIZES[@]}"
