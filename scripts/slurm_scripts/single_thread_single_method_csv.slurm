#!/bin/bash
#SBATCH --job-name=st_one_method
#SBATCH --account=MPHIL-DIS-SL2-CPU
#SBATCH --partition=sapphire
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

ROOT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${ROOT_DIR}"

OPTIMISATION_METHOD="${1:-baseline}"
MATRIX_SIZE="${2:-50000}"
OUTPUT_CSV="${3:-results/raw/single_thread_single_method/${OPTIMISATION_METHOD}_n${MATRIX_SIZE}.csv}"

mkdir -p "$(dirname "${OUTPUT_CSV}")"

module purge
module load cmake
module load gcc

bash scripts/sh_scripts/build.sh

./build/run/run_cholesky \
    time \
    "${OPTIMISATION_METHOD}" \
    "${MATRIX_SIZE}" \
    "${OUTPUT_CSV}"
