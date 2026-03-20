#!/bin/bash
#SBATCH --job-name=st_matrix_sizes_all
#SBATCH --account=MPHIL-DIS-SL2-CPU
#SBATCH --partition=icelake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=02:30:00
#SBATCH --output=logs/single_thread/slurm-%x-%j.out
#SBATCH --error=logs/single_thread/slurm-%x-%j.err

set -euo pipefail

. /etc/profile.d/modules.sh

# Clean build environment.
module purge
module load rhel8/default-icl
module load cmake
module load gcc

# Move to project root.
ROOT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${ROOT_DIR}"

# Run each case 10 times unless overridden.
REPEATS="${REPEATS:-10}"

# Use one thread for everything, including the OpenMP methods.
THREADS=1

# Block sizes for blocked methods.
SERIAL_BLOCK_SIZE=32
TILE_PARALLEL_BLOCK_SIZE=32
BLOCK_ROW_BLOCK_SIZE=64
TILE_LIST_BLOCK_SIZE=32
TASK_DAG_BLOCK_SIZE=64

# All methods except baseline.
METHODS=(
    lower_triangle
    upper_triangle
    contiguous_access
    cholesky_blocked_tile_kernels
    cholesky_blocked_tile_kernels_unrolled
    openmp_row_parallel_unblocked
    openmp_tile_parallel_blocked
    openmp_block_row_parallel
    openmp_tile_list_parallel
    openmp_task_dag_blocked
)

# Sweep matrix sizes from 1 to 5000.
SIZES=()
for ((n = 1; n <= 5000; n += 200)); do
    SIZES+=("${n}")
done

RAW_DIR="results/raw/slurm/single_thread_matrix_size_compare_all"
PLOT_DIR="results/figures/slurm/single_thread_matrix_size_compare_all"
RAW_CSVS=()

mkdir -p "${RAW_DIR}" "${PLOT_DIR}" logs/single_thread

echo "== Job info =="
echo "HOSTNAME=${HOSTNAME}"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-}"
echo "THREADS=${THREADS}"
echo "REPEATS=${REPEATS}"
echo "METHODS=${METHODS[*]}"
echo "SIZE_RANGE=1..5000"

echo "== Building =="
cmake --preset linux-release-no-tests
cmake --build --preset linux-benchmark-no-tests --parallel

echo "== OpenMP linkage =="
ldd ./build/run/run_cholesky | grep -E 'gomp|iomp|omp' || true

for METHOD in "${METHODS[@]}"; do
    RAW_CSV="${RAW_DIR}/${METHOD}.csv"
    RAW_CSVS+=("${RAW_CSV}")

    echo "== Running method=${METHOD} =="

    EXTRA_ARGS=()

    case "${METHOD}" in
        cholesky_blocked_tile_kernels|cholesky_blocked_tile_kernels_unrolled)
            EXTRA_ARGS=(--block-size "${SERIAL_BLOCK_SIZE}")
            ;;
        openmp_row_parallel_unblocked)
            EXTRA_ARGS=(--threads "${THREADS}")
            ;;
        openmp_tile_parallel_blocked)
            EXTRA_ARGS=(--threads "${THREADS}" --block-size "${TILE_PARALLEL_BLOCK_SIZE}")
            ;;
        openmp_block_row_parallel)
            EXTRA_ARGS=(--threads "${THREADS}" --block-size "${BLOCK_ROW_BLOCK_SIZE}")
            ;;
        openmp_tile_list_parallel)
            EXTRA_ARGS=(--threads "${THREADS}" --block-size "${TILE_LIST_BLOCK_SIZE}")
            ;;
        openmp_task_dag_blocked)
            EXTRA_ARGS=(--threads "${THREADS}" --block-size "${TASK_DAG_BLOCK_SIZE}")
            ;;
    esac

    srun --cpu-bind=cores ./build/run/run_cholesky \
        matrix-size-sweep \
        "${METHOD}" \
        "${REPEATS}" \
        "${RAW_CSV}" \
        "${SIZES[@]}" \
        "${EXTRA_ARGS[@]}"
done

module load python
source ~/venvs/c2_coursework/bin/activate

MPLCONFIGDIR="${TMPDIR:-/tmp}/mpl_${SLURM_JOB_ID}" \
python plot/cholesky_plotter.py \
    matrix-size-comparison \
    "${PLOT_DIR}" \
    "${RAW_CSVS[@]}"