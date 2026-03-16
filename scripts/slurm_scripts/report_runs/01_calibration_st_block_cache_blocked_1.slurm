#!/bin/bash
#SBATCH --job-name=cal_st_cb1
#SBATCH --account=MPHIL-DIS-SL2-CPU
#SBATCH --partition=icelake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:45:00
#SBATCH --output=logs/single_thread/block_sweep/slurm-%x-%j.out
#SBATCH --error=logs/single_thread/block_sweep/slurm-%x-%j.err

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

bash scripts/sh_scripts/run_block_size_sweep.sh \
    cache_blocked_1 \
    2000 \
    10 \
    {1..256}
