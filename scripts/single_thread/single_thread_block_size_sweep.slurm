#!/bin/bash
#SBATCH --job-name=st_block_sweep
#SBATCH --account=MPHIL-DIS-SL2-CPU
#SBATCH --partition=icelake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00
#SBATCH --output=logs/single_thread/slurm-%x-%j.out
#SBATCH --error=logs/single_thread/slurm-%x-%j.err

set -euo pipefail

. /etc/profile.d/modules.sh

# Load a clean build environment.
module purge
module load rhel8/default-icl
module load cmake
module load gcc

# Move to the project root.
ROOT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${ROOT_DIR}"

# Benchmark settings.
MATRIX_SIZE="${MATRIX_SIZE:-2000}"
REPEATS="${REPEATS:-3}"

# Single-thread blocked methods to compare.
METHODS=(
    cholesky_blocked_tile_kernels
    cholesky_blocked_tile_kernels_unrolled
)

# Sweep block sizes from 1 to 128.
BLOCK_SIZES=($(seq 1 128))

# Output locations.
RAW_DIR="results/raw/slurm/single_thread_block_size_sweep"
PLOT_DIR="results/figures/slurm/single_thread_block_size_sweep_n${MATRIX_SIZE}"
COMBINED_RAW_CSV="${RAW_DIR}/single_thread_block_size_sweep_n${MATRIX_SIZE}.csv"
COMBINED_SUMMARY_CSV="${RAW_DIR}/single_thread_block_size_sweep_n${MATRIX_SIZE}_summary.csv"

# Keep track of the per-method CSV files so they can be merged later.
RAW_CSVS=()
SUMMARY_CSVS=()

mkdir -p "${RAW_DIR}" "${PLOT_DIR}" logs/single_thread

echo "== Job info =="
echo "HOSTNAME=${HOSTNAME}"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-}"
echo "MATRIX_SIZE=${MATRIX_SIZE}"
echo "REPEATS=${REPEATS}"

# Build once before running the benchmarks.
echo "== Building =="
cmake --preset linux-release-no-tests
cmake --build --preset linux-benchmark-no-tests --parallel

# Run one block-size sweep per method.
for METHOD in "${METHODS[@]}"; do
    RAW_CSV="${RAW_DIR}/${METHOD}_n${MATRIX_SIZE}.csv"
    SUMMARY_CSV="${RAW_CSV%.csv}_summary.csv"

    RAW_CSVS+=("${RAW_CSV}")
    SUMMARY_CSVS+=("${SUMMARY_CSV}")

    echo "== Running method=${METHOD}, n=${MATRIX_SIZE} =="

    srun --cpu-bind=cores ./build/run/run_cholesky \
        block-size-sweep \
        "${METHOD}" \
        "${MATRIX_SIZE}" \
        "${REPEATS}" \
        "${RAW_CSV}" \
        "${BLOCK_SIZES[@]}"
done

# Merge the raw CSV files into one combined file.
# Keep the header from the first file only.
awk 'FNR == 1 && NR != 1 { next } { print }' "${RAW_CSVS[@]}" > "${COMBINED_RAW_CSV}"

# Merge the summary CSV files in the same way.
awk 'FNR == 1 && NR != 1 { next } { print }' "${SUMMARY_CSVS[@]}" > "${COMBINED_SUMMARY_CSV}"

# Activate Python and generate the plots.
module load python
source ~/venvs/c2_coursework/bin/activate

MPLCONFIGDIR="${TMPDIR:-/tmp}/mpl_${SLURM_JOB_ID}" \
python plot/cholesky_plotter.py \
    block-size \
    "${COMBINED_RAW_CSV}" \
    "${PLOT_DIR}"