#!/bin/bash
#SBATCH --job-name=st_fix_20000
#SBATCH --account=MPHIL-DIS-SL2-CPU
#SBATCH --partition=icelake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=13G
#SBATCH --time=05:00:00
#SBATCH --output=logs/single_thread/fixedSize/slurm-%x-%j.out
#SBATCH --error=logs/single_thread/fixedSize/slurm-%x-%j.err

set -euo pipefail

. /etc/profile.d/modules.sh

ROOT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${ROOT_DIR}"

mkdir -p results/raw/report_runs

module purge
module load rhel8/default-icl
module load cmake
module load gcc

export MPHIL_CACHE_BLOCKED_1_BLOCK_SIZE="${MPHIL_CACHE_BLOCKED_1_BLOCK_SIZE:-32}"
export MPHIL_CACHE_BLOCKED_2_BLOCK_SIZE="${MPHIL_CACHE_BLOCKED_2_BLOCK_SIZE:-32}"

bash scripts/sh_scripts/build.sh

./build/run/run_cholesky \
    fixed-size \
    20000 \
    3 \
    results/raw/report_runs/fixed_size_comparison_n20000_raw.csv \
    lower_triangle upper_triangle contiguous_access cache_blocked_1 cache_blocked_2
