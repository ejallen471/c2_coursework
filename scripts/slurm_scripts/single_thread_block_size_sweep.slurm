#!/bin/bash
#SBATCH --job-name=st_block_sweep
#SBATCH --account=MPHIL-DIS-SL2-CPU
#SBATCH --partition=sapphire
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

ROOT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${ROOT_DIR}"

OPTIMISATION_METHOD="${1:-cache_blocked_1}"
MATRIX_SIZE="${2:-2000}"
REPEATS="${3:-5}"

shift $(( $# >= 3 ? 3 : $# ))

if [ "$#" -eq 0 ]; then
    BLOCK_SIZES=(8 12 16 24 32 40 48 64 80 96 128)
else
    BLOCK_SIZES=("$@")
fi

RAW_CSV="results/raw/${OPTIMISATION_METHOD}_block_size_sweep_n${MATRIX_SIZE}.csv"
PLOT_DIR="results/figures/${OPTIMISATION_METHOD}_block_size_sweep_n${MATRIX_SIZE}"

mkdir -p results/raw results/figures

module purge
module load cmake
module load gcc
module load python

source ~/venvs/c2_coursework/bin/activate

bash scripts/sh_scripts/build.sh

./build/run/run_cholesky \
    block-size-sweep \
    "${OPTIMISATION_METHOD}" \
    "${MATRIX_SIZE}" \
    "${REPEATS}" \
    "${RAW_CSV}" \
    "${BLOCK_SIZES[@]}"

MPLCONFIGDIR="${TMPDIR:-/tmp}/mpl_${SLURM_JOB_ID}" \
python plot/plot_block_size_metrics.py \
    "${RAW_CSV}" \
    "${PLOT_DIR}"

echo "raw_csv=\"${RAW_CSV}\""
echo "plot_dir=\"${PLOT_DIR}\""
