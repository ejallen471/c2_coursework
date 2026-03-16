#!/bin/bash
#SBATCH --job-name=cal_omp_thr
#SBATCH --account=MPHIL-DIS-SL2-CPU
#SBATCH --partition=icelake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=76
#SBATCH --time=12:00:00
#SBATCH --output=logs/openMP/thread_sweep/slurm-%x-%j.out
#SBATCH --error=logs/openMP/thread_sweep/slurm-%x-%j.err

set -euo pipefail

. /etc/profile.d/modules.sh

ROOT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${ROOT_DIR}"

mkdir -p logs/openMP/thread_sweep

module purge
module load rhel8/default-icl
module load cmake
module load gcc
module load python

source ~/venvs/c2_coursework/bin/activate

rm -rf build_openmp
bash scripts/sh_scripts/build_openmp.sh

bash scripts/sh_scripts/run_openmp_thread_count_sweep.sh \
    10000 \
    5 \
    {1..76}
