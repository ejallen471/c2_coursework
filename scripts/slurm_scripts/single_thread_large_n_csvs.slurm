#!/bin/bash
#SBATCH --job-name=st_large_n
#SBATCH --account=MPHIL-DIS-SL2-CPU
#SBATCH --partition=sapphire
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=220G
#SBATCH --time=72:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

ROOT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${ROOT_DIR}"

if [ "$#" -eq 0 ]; then
    SIZES=(50000 100000)
else
    SIZES=("$@")
fi

METHODS=(baseline lower_triangle upper_triangle contiguous_access cache_blocked_1 cache_blocked_2)
OUTPUT_DIR="results/raw/single_thread_large_n"

mkdir -p "${OUTPUT_DIR}"

module purge
module load cmake
module load gcc

bash scripts/sh_scripts/build.sh

for MATRIX_SIZE in "${SIZES[@]}"; do
    for METHOD in "${METHODS[@]}"; do
        OUTPUT_CSV="${OUTPUT_DIR}/${METHOD}_n${MATRIX_SIZE}.csv"
        echo "==> Running ${METHOD} for n=${MATRIX_SIZE}"
        ./build/run/run_cholesky \
            time \
            "${METHOD}" \
            "${MATRIX_SIZE}" \
            "${OUTPUT_CSV}"
    done
done
