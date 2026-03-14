#!/bin/bash
#SBATCH --job-name=omp_threads
#SBATCH --account=MPHIL-DIS-SL2-CPU
#SBATCH --partition=sapphire
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

ROOT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${ROOT_DIR}"

MATRIX_SIZE="${1:-2000}"
REPEATS="${2:-5}"

shift $(( $# >= 2 ? 2 : $# ))

if [ "$#" -eq 0 ]; then
    THREAD_COUNTS=(1 2 4 8 16)
else
    THREAD_COUNTS=("$@")
fi

module purge
module load cmake
module load gcc
module load python

source ~/venvs/c2_coursework/bin/activate

bash scripts/sh_scripts/run_openmp_thread_count_sweep.sh \
    "${MATRIX_SIZE}" \
    "${REPEATS}" \
    "${THREAD_COUNTS[@]}"
