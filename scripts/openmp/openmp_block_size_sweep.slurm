#!/bin/bash
#SBATCH --job-name=omp_block_sweep
#SBATCH --account=MPHIL-DIS-SL2-CPU
#SBATCH --partition=icelake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=76
#SBATCH --hint=nomultithread
#SBATCH --time=02:00:00
#SBATCH --output=logs/openmp/slurm-%x-%j.out
#SBATCH --error=logs/openmp/slurm-%x-%j.err

set -euo pipefail

. /etc/profile.d/modules.sh

module purge
module load rhel8/default-icl
module load cmake
module load gcc

# Move to the project root.
ROOT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${ROOT_DIR}"

# Benchmark settings.
MATRIX_SIZE="${MATRIX_SIZE:-5000}"
REPEATS="${REPEATS:-3}"

# Methods included in this sweep.
METHODS=(
    openmp_tile_parallel_blocked
    openmp_block_row_parallel
    openmp_tile_list_parallel
    openmp_task_dag_blocked
)

# Try block sizes from 1 to 200.
BLOCK_SIZES=($(seq 1 200))

# Output locations.
RAW_DIR="results/raw/slurm/openmp_block_size_sweep"
PLOT_DIR="results/figures/slurm/openmp_block_size_sweep_n${MATRIX_SIZE}"
COMBINED_RAW_CSV="${RAW_DIR}/openmp_block_size_sweep_n${MATRIX_SIZE}.csv"
COMBINED_SUMMARY_CSV="${RAW_DIR}/openmp_block_size_sweep_n${MATRIX_SIZE}_summary.csv"

# Store the per-method CSV paths so they can be merged later.
RAW_CSVS=()
SUMMARY_CSVS=()

mkdir -p "${RAW_DIR}" "${PLOT_DIR}" logs/openmp

echo "== Job info =="
echo "HOSTNAME=${HOSTNAME}"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-}"
echo "MATRIX_SIZE=${MATRIX_SIZE}"
echo "REPEATS=${REPEATS}"

# Build the project once before running any benchmarks.
cmake --preset linux-release-no-tests
cmake --build --preset linux-benchmark-no-tests --parallel

# Run one block-size sweep per method.
for METHOD in "${METHODS[@]}"; do
    RAW_CSV="${RAW_DIR}/${METHOD}_n${MATRIX_SIZE}.csv"
    SUMMARY_CSV="${RAW_CSV%.csv}_summary.csv"

    RAW_CSVS+=("${RAW_CSV}")
    SUMMARY_CSVS+=("${SUMMARY_CSV}")

    echo "== Running ${METHOD} =="
    echo "Output: ${RAW_CSV}"

    srun --cpu-bind=cores ./build/run/run_cholesky \
        block-size-sweep \
        "${METHOD}" \
        "${MATRIX_SIZE}" \
        "${REPEATS}" \
        "${RAW_CSV}" \
        "${BLOCK_SIZES[@]}" \
        --threads "${SLURM_CPUS_PER_TASK}"
done

# Merge all raw CSV files into one combined file.
# Keep the header from the first file only.
awk 'FNR == 1 && NR != 1 { next } { print }' "${RAW_CSVS[@]}" > "${COMBINED_RAW_CSV}"

# Merge all summary CSV files in the same way.
awk 'FNR == 1 && NR != 1 { next } { print }' "${SUMMARY_CSVS[@]}" > "${COMBINED_SUMMARY_CSV}"

# Activate the Python environment for plotting.
module load python
source ~/venvs/c2_coursework/bin/activate

# Use a writable matplotlib config directory on the compute node.
MPLCONFIGDIR="${TMPDIR:-/tmp}/mpl_${SLURM_JOB_ID}" \
python plot/cholesky_plotter.py \
    block-size \
    "${COMBINED_RAW_CSV}" \
    "${PLOT_DIR}"
