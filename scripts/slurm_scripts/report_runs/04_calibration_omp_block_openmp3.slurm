#!/bin/bash
#SBATCH --job-name=cal_omp_blk3
#SBATCH --account=MPHIL-DIS-SL2-CPU
#SBATCH --partition=icelake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=06:00:00
#SBATCH --output=logs/openMP/block_sweep/slurm-%x-%j.out
#SBATCH --error=logs/openMP/block_sweep/slurm-%x-%j.err

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

bash scripts/sh_scripts/run_openmp_block_size_sweep.sh \
    openmp3 \
    10000 \
    10 \
    {1..256}
