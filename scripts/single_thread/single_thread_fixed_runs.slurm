#!/bin/bash
#SBATCH --job-name=st_fixed_runs
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
REPEATS="${REPEATS:-3}"
BLOCK_SIZE=32

# Matrix sizes to test.
SIZES=(2000 5000 10000)

# Single-thread methods to compare.
METHODS=(
    baseline
    lower_triangle
    upper_triangle
    contiguous_access
    cholesky_blocked_tile_kernels
    cholesky_blocked_tile_kernels_unrolled
)

# Output location.
RAW_DIR="results/raw/slurm/single_thread_fixed_runs"
mkdir -p "${RAW_DIR}" logs/single_thread

echo "== Job info =="
echo "HOSTNAME=${HOSTNAME}"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-}"
echo "REPEATS=${REPEATS}"
echo "BLOCK_SIZE=${BLOCK_SIZE}"
echo "SIZES=${SIZES[*]}"

# Build once before running any benchmarks.
echo "== Building =="
cmake --preset linux-release-no-tests
cmake --build --preset linux-benchmark-no-tests --parallel

# Run the fixed-size comparison for each matrix size.
for MATRIX_SIZE in "${SIZES[@]}"; do
    RAW_CSV="${RAW_DIR}/single_thread_method_compare_n${MATRIX_SIZE}.csv"

    echo "== Running n=${MATRIX_SIZE} =="
    echo "Methods: ${METHODS[*]}"
    echo "Output: ${RAW_CSV}"

    srun --cpu-bind=cores ./build/run/run_cholesky \
        method-compare \
        "${MATRIX_SIZE}" \
        "${REPEATS}" \
        "${RAW_CSV}" \
        "${METHODS[@]}" \
        --block-size "${BLOCK_SIZE}"
done