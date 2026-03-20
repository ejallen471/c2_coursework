#!/bin/bash
#SBATCH --job-name=omp_threads
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

# Load a clean environment.
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

# Methods to include in the thread-count sweep.
METHODS=(
    openmp_task_dag_blocked
    # openmp_row_parallel_unblocked
    # openmp_tile_parallel_blocked
    # openmp_block_row_parallel
    # openmp_tile_list_parallel  
)

# Sweep from 1 thread up to the full Slurm allocation.
THREAD_COUNTS=()
for ((threads = 1; threads <= SLURM_CPUS_PER_TASK; ++threads)); do
    THREAD_COUNTS+=("${threads}")
done

# Output locations.
RAW_DIR="results/raw/slurm/openmp_thread_count_sweep"
RAW_CSV="${RAW_DIR}/openmp_thread_count_sweep_n${MATRIX_SIZE}.csv"
PLOT_DIR="results/figures/slurm/openmp_thread_count_sweep_n${MATRIX_SIZE}"

mkdir -p "${RAW_DIR}" "${PLOT_DIR}" logs/openmp

# Shared block size used for blocked methods in this sweep.
THREAD_COUNT_SWEEP_BLOCK_SIZE=32

echo "== Job info =="
echo "HOSTNAME=${HOSTNAME}"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-}"
echo "MATRIX_SIZE=${MATRIX_SIZE}"
echo "REPEATS=${REPEATS}"
echo "THREAD_COUNT_SWEEP_BLOCK_SIZE=${THREAD_COUNT_SWEEP_BLOCK_SIZE}"

# Build once before running benchmarks.
echo "== Building =="
cmake --preset linux-release-no-tests
cmake --build --preset linux-benchmark-no-tests --parallel

# Quick check that the benchmark binary is linked against an OpenMP runtime.
echo "== OpenMP linkage =="
ldd ./build/run/run_cholesky | grep -E 'gomp|iomp|omp' || true

# Run the full thread-count sweep.
srun --cpu-bind=cores ./build/run/run_cholesky \
    thread-count-sweep \
    "${MATRIX_SIZE}" \
    "${REPEATS}" \
    "${RAW_CSV}" \
    --threads "${THREAD_COUNTS[@]}" \
    --block-size "${THREAD_COUNT_SWEEP_BLOCK_SIZE}" \
    --methods "${METHODS[@]}"

# Activate Python and generate plots.
module load python
source ~/venvs/c2_coursework/bin/activate

MPLCONFIGDIR="${TMPDIR:-/tmp}/mpl_${SLURM_JOB_ID}" \
python plot/cholesky_plotter.py \
    thread-count \
    "${RAW_CSV}" \
    "${PLOT_DIR}"