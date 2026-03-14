#!/bin/bash
#SBATCH --job-name=omp_fixed_size
#SBATCH --account=MPHIL-DIS-SL2-CPU
#SBATCH --partition=sapphire
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

ROOT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${ROOT_DIR}"

MATRIX_SIZE="${1:-2000}"
REPEATS="${2:-1}"
OUTPUT_CSV="${3:-results/raw/openmp_fixed_size_comparison_n${MATRIX_SIZE}.csv}"

module purge
module load cmake
module load gcc
module load python

source ~/venvs/c2_coursework/bin/activate

export OPENMP_THREADS="${OPENMP_THREADS:-${SLURM_CPUS_PER_TASK}}"

bash scripts/sh_scripts/run_openmp_fixed_size_comparison.sh \
    "${MATRIX_SIZE}" \
    "${REPEATS}" \
    "${OUTPUT_CSV}"
