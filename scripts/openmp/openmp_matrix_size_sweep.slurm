#!/bin/bash
#SBATCH --job-name=omp_matrix_sizes
#SBATCH --account=MPHIL-DIS-SL2-CPU
#SBATCH --partition=icelake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=76
#SBATCH --hint=nomultithread
#SBATCH --time=05:00:00
#SBATCH --output=logs/openmp/slurm-%x-%j.out
#SBATCH --error=logs/openmp/slurm-%x-%j.err

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

# How many times to repeat each run.
REPEATS="${REPEATS:-3}"

# Methods to compare across matrix sizes.
METHODS=(
    openmp_tile_parallel_blocked
    openmp_block_row_parallel
    openmp_tile_list_parallel
    openmp_task_dag_blocked
)

# Matrix sizes to test.
# Keep the small 100 case, then sweep from 5000 to 100000.
SIZES=(100)
for ((n = 5000; n <= 100000; n += 5000)); do
    SIZES+=("${n}")
done

# Output locations.
RAW_DIR="results/raw/slurm/openmp_matrix_size_compare"
PLOT_DIR="results/figures/slurm/openmp_matrix_size_compare"
RAW_CSVS=()

mkdir -p "${RAW_DIR}" "${PLOT_DIR}" logs/openmp

# Fixed block sizes for each method.
TILE_PARALLEL_BLOCK_SIZE=32
BLOCK_ROW_BLOCK_SIZE=64
TILE_LIST_BLOCK_SIZE=32
TASK_DAG_BLOCK_SIZE=64

echo "== Job info =="
echo "HOSTNAME=${HOSTNAME}"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-}"
echo "REPEATS=${REPEATS}"
echo "METHODS=${METHODS[*]}"

# Build once before running benchmarks.
echo "== Building =="
cmake --preset linux-release-no-tests
cmake --build --preset linux-benchmark-no-tests --parallel

# Quick OpenMP check.
echo "== OpenMP linkage =="
ldd ./build/run/run_cholesky | grep -E 'gomp|iomp|omp' || true

# Run one matrix-size sweep per method.
for METHOD in "${METHODS[@]}"; do
    RAW_CSV="${RAW_DIR}/${METHOD}.csv"
    RAW_CSVS+=("${RAW_CSV}")

    echo "== Running method=${METHOD} =="

    # Pick the block size for this method.
    BLOCK_SIZE=""
    case "${METHOD}" in
        openmp_tile_parallel_blocked)
            BLOCK_SIZE="${TILE_PARALLEL_BLOCK_SIZE}"
            ;;
        openmp_block_row_parallel)
            BLOCK_SIZE="${BLOCK_ROW_BLOCK_SIZE}"
            ;;
        openmp_tile_list_parallel)
            BLOCK_SIZE="${TILE_LIST_BLOCK_SIZE}"
            ;;
        openmp_task_dag_blocked)
            BLOCK_SIZE="${TASK_DAG_BLOCK_SIZE}"
            ;;
    esac

    srun --cpu-bind=cores ./build/run/run_cholesky \
        matrix-size-sweep \
        "${METHOD}" \
        "${REPEATS}" \
        "${RAW_CSV}" \
        "${SIZES[@]}" \
        --threads "${SLURM_CPUS_PER_TASK}" \
        --block-size "${BLOCK_SIZE}"
done

# Load Python and make the comparison plots.
module load python
source ~/venvs/c2_coursework/bin/activate

MPLCONFIGDIR="${TMPDIR:-/tmp}/mpl_${SLURM_JOB_ID}" \
python plot/cholesky_plotter.py \
    matrix-size-comparison \
    "${PLOT_DIR}" \
    "${RAW_CSVS[@]}"