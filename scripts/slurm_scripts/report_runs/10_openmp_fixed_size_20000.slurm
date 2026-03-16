#!/bin/bash
#SBATCH --job-name=omp_fix_20000
#SBATCH --account=MPHIL-DIS-SL2-CPU
#SBATCH --partition=icelake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=04:00:00
#SBATCH --output=logs/openMP/fixedSize/slurm-%x-%j.out
#SBATCH --error=logs/openMP/fixedSize/slurm-%x-%j.err

set -euo pipefail

. /etc/profile.d/modules.sh

ROOT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${ROOT_DIR}"

module purge
module load rhel8/default-icl
module load cmake
module load gcc
module load python

source ~/venvs/c2_coursework/bin/activate

export OPENMP_THREADS="${OPENMP_THREADS:-32}"
export MPHIL_OPENMP3_BLOCK_SIZE="${MPHIL_OPENMP3_BLOCK_SIZE:-32}"
export MPHIL_OPENMP4_BLOCK_SIZE="${MPHIL_OPENMP4_BLOCK_SIZE:-32}"

bash scripts/sh_scripts/run_openmp_fixed_size_comparison.sh \
    20000 \
    3 \
    results/raw/report_runs/openmp_fixed_size_comparison_n20000.csv \
    --methods openmp1 openmp2 openmp3 openmp4
