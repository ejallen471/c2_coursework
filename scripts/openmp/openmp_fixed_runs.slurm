#!/bin/bash
#SBATCH --job-name=omp_fixed_runs
#SBATCH --account=MPHIL-DIS-SL2-CPU
#SBATCH --partition=icelake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=76
#SBATCH --hint=nomultithread
#SBATCH --time=00:10:00
#SBATCH --output=logs/openmp/slurm-%x-%j.out
#SBATCH --error=logs/openmp/slurm-%x-%j.err

set -euo pipefail

. /etc/profile.d/modules.sh

# Load a clean environment.
module purge
module load rhel8/default-icl
module load cmake
module load gcc

# Move to project root.
ROOT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${ROOT_DIR}"

# Basic run settings.
REPEATS="${REPEATS:-1}"
SIZES=(5000)

RAW_DIR="results/raw/slurm/openmp_fixed_runs"
mkdir -p "${RAW_DIR}" logs/openmp

# Block sizes for each method.
TILE_PARALLEL_BLOCK_SIZE=32
BLOCK_ROW_BLOCK_SIZE=64
TILE_LIST_BLOCK_SIZE=32
TASK_DAG_BLOCK_SIZE=64

echo "== Job info =="
echo "HOSTNAME=${HOSTNAME}"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-}"
echo "REPEATS=${REPEATS}"
echo "SIZES=${SIZES[*]}"

# Sanity check.
if [ -z "${SLURM_CPUS_PER_TASK:-}" ]; then
    echo "Error: SLURM_CPUS_PER_TASK not set"
    exit 1
fi

# Build the project.
echo "== Building =="
cmake --preset linux-release-no-tests
cmake --build --preset linux-benchmark-no-tests --parallel

# Check OpenMP is actually linked (important for your issue).
echo "== Checking OpenMP linkage =="
if ! ldd ./build/run/run_cholesky | grep -E 'gomp|iomp|omp'; then
    echo "Error: OpenMP runtime not linked"
    exit 1
fi

# Loop over matrix sizes.
for MATRIX_SIZE in "${SIZES[@]}"; do

    # Choose methods.
    # For large matrices, skip the slow/unblocked version.
    METHODS=(
        openmp_tile_parallel_blocked
        openmp_block_row_parallel
        openmp_tile_list_parallel
        openmp_task_dag_blocked
    )

    if [ "${MATRIX_SIZE}" -lt 50000 ]; then
        METHODS=(
            openmp_row_parallel_unblocked
            "${METHODS[@]}"
        )
    fi

    RAW_CSV="${RAW_DIR}/openmp_method_compare_n${MATRIX_SIZE}.csv"

    echo "== Running n=${MATRIX_SIZE} =="
    echo "Methods: ${METHODS[*]}"
    echo "Output: ${RAW_CSV}"

    # Run benchmark.
    srun --cpu-bind=cores ./build/run/run_cholesky \
        method-compare \
        "${MATRIX_SIZE}" \
        "${REPEATS}" \
        "${RAW_CSV}" \
        "${METHODS[@]}" \
        --threads "${SLURM_CPUS_PER_TASK}" \
        --block-size-for "openmp_tile_parallel_blocked=${TILE_PARALLEL_BLOCK_SIZE}" \
        --block-size-for "openmp_block_row_parallel=${BLOCK_ROW_BLOCK_SIZE}" \
        --block-size-for "openmp_tile_list_parallel=${TILE_LIST_BLOCK_SIZE}" \
        --block-size-for "openmp_task_dag_blocked=${TASK_DAG_BLOCK_SIZE}"

done