#!/bin/bash
#SBATCH --job-name=cal_omp_blk4
#SBATCH --account=MPHIL-DIS-SL2-CPU
#SBATCH --partition=icelake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=76
#SBATCH --time=02:00:00
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

export OPENMP_THREADS="${OPENMP_THREADS:-76}" #! The number of threads this is tested with

bash scripts/sh_scripts/run_openmp_block_size_sweep.sh \
    openmp4 \
    10000 \
    10 \
    {1..128}
