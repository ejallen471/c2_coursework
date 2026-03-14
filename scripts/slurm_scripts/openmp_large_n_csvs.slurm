#!/bin/bash
#SBATCH --job-name=omp_large_n
#SBATCH --account=MPHIL-DIS-SL2-CPU
#SBATCH --partition=sapphire
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=220G
#SBATCH --time=72:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

ROOT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${ROOT_DIR}"

if [ "$#" -eq 0 ]; then
    SIZES=(100000)
else
    SIZES=("$@")
fi

METHODS=(baseline openmp1 openmp2 openmp3 openmp4)
OUTPUT_DIR="results/raw/openmp_large_n"
OPENMP_THREADS="${OPENMP_THREADS:-${SLURM_CPUS_PER_TASK}}"

mkdir -p "${OUTPUT_DIR}"

module purge
module load cmake
module load gcc
module load python

source ~/venvs/c2_coursework/bin/activate

bash scripts/sh_scripts/build_openmp.sh

for MATRIX_SIZE in "${SIZES[@]}"; do
    for METHOD in "${METHODS[@]}"; do
        OUTPUT_CSV="${OUTPUT_DIR}/${METHOD}_n${MATRIX_SIZE}.csv"
        echo "==> Running ${METHOD} for n=${MATRIX_SIZE}"

        if [ "${METHOD}" = "baseline" ]; then
            OMP_NUM_THREADS=1 ./build_openmp/run/run_cholesky \
                time \
                "${METHOD}" \
                "${MATRIX_SIZE}" \
                "${OUTPUT_CSV}"
        else
            OMP_NUM_THREADS="${OPENMP_THREADS}" ./build_openmp/run/run_cholesky \
                time \
                "${METHOD}" \
                "${MATRIX_SIZE}" \
                "${OUTPUT_CSV}"
        fi
    done
done
