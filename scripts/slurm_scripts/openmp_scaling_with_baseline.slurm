#!/bin/bash
#SBATCH --job-name=omp_scaling
#SBATCH --account=MPHIL-DIS-SL2-CPU
#SBATCH --partition=sapphire
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=08:00:00
#SBATCH --output=logs/openMP_scaling/slurm-%x-%j.out
#SBATCH --error=logs/openMP_scaling/slurm-%x-%j.err

set -euo pipefail

ROOT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${ROOT_DIR}"

REPEATS="${1:-5}"

shift $(( $# >= 1 ? 1 : $# ))

if [ "$#" -eq 0 ]; then
    SIZES=(512 1024 2000 4096 8192)
else
    SIZES=("$@")
fi

module purge
module load cmake
module load gcc
module load python

source ~/venvs/c2_coursework/bin/activate

export OPENMP_THREADS="${OPENMP_THREADS:-${SLURM_CPUS_PER_TASK}}"

bash scripts/sh_scripts/run_openmp_methods_graph.sh "${REPEATS}" "${SIZES[@]}"
