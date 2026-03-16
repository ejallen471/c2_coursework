#!/bin/bash
#SBATCH --job-name=omp4_n100000
#SBATCH --account=MPHIL-DIS-SL2-CPU
#SBATCH --partition=icelake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=56
#SBATCH --mem=256G
#SBATCH --time=01:00:00
#SBATCH --output=logs/openMP/largeMatrix/slurm-%x-%j.out
#SBATCH --error=logs/openMP/largeMatrix/slurm-%x-%j.err

set -euo pipefail

. /etc/profile.d/modules.sh

ROOT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${ROOT_DIR}"

mkdir -p results/raw/report_runs

module purge
module load rhel8/default-icl
module load cmake
module load gcc

export OPENMP_THREADS="${OPENMP_THREADS:-56}"
export MPHIL_OPENMP4_BLOCK_SIZE="${MPHIL_OPENMP4_BLOCK_SIZE:-32}"

bash scripts/sh_scripts/build_openmp.sh

OMP_NUM_THREADS="${OPENMP_THREADS}" \
./build_openmp/run/run_cholesky \
    time \
    openmp4 \
    100000 \
    results/raw/report_runs/openmp4_n100000.csv
