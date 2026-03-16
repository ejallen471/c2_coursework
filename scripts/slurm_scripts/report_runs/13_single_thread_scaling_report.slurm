#!/bin/bash
#SBATCH --job-name=st_scaling_rpt
#SBATCH --account=MPHIL-DIS-SL2-CPU
#SBATCH --partition=icelake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=05:00:00
#SBATCH --output=logs/single_thread/scaling/slurm-%x-%j.out
#SBATCH --error=logs/single_thread/scaling/slurm-%x-%j.err

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

export MPHIL_CACHE_BLOCKED_1_BLOCK_SIZE="${MPHIL_CACHE_BLOCKED_1_BLOCK_SIZE:-32}"
export MPHIL_CACHE_BLOCKED_2_BLOCK_SIZE="${MPHIL_CACHE_BLOCKED_2_BLOCK_SIZE:-32}"

bash scripts/sh_scripts/run_all_methods_graph.sh \
    3 \
    --methods \
    lower_triangle \
    upper_triangle \
    contiguous_access \
    cache_blocked_1 \
    cache_blocked_2 \
    --sizes \
    256 384 512 768 1000 1500 2000 3000 4000 6000 8000 12000 16000 20000
