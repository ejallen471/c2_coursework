#!/bin/bash
#SBATCH --job-name=st_matrix_sizes
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

# Single-thread methods to compare across matrix sizes.
METHODS=(
    baseline
    lower_triangle
    upper_triangle
    contiguous_access
    cholesky_blocked_tile_kernels
    cholesky_blocked_tile_kernels_unrolled
)

# Matrix sizes to test.
# Start with a tiny case, then sweep up to 10000.
SIZES=(100)
for ((n = 500; n <= 10000; n += 500)); do
    SIZES+=("${n}")
done

# Output locations.
RAW_DIR="results/raw/slurm/single_thread_matrix_size_compare"
PLOT_DIR="results/figures/slurm/single_thread_matrix_size_compare"
RAW_CSVS=()

mkdir -p "${RAW_DIR}" "${PLOT_DIR}" logs/single_thread

echo "== Job info =="
echo "HOSTNAME=${HOSTNAME}"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-}"
echo "REPEATS=${REPEATS}"
echo "BLOCK_SIZE=${BLOCK_SIZE}"
echo "METHODS=${METHODS[*]}"

# Build once before running benchmarks.
echo "== Building =="
cmake --preset linux-release-no-tests
cmake --build --preset linux-benchmark-no-tests --parallel

# Run one matrix-size sweep per method.
for METHOD in "${METHODS[@]}"; do
    RAW_CSV="${RAW_DIR}/${METHOD}.csv"
    RAW_CSVS+=("${RAW_CSV}")

    echo "== Running method=${METHOD} =="

    # Only blocked methods need a block-size argument.
    EXTRA_ARGS=()
    case "${METHOD}" in
        cholesky_blocked_tile_kernels|cholesky_blocked_tile_kernels_unrolled)
            EXTRA_ARGS=(--block-size "${BLOCK_SIZE}")
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

# Activate Python and generate the comparison plots.
module load python
source ~/venvs/c2_coursework/bin/activate

MPLCONFIGDIR="${TMPDIR:-/tmp}/mpl_${SLURM_JOB_ID}" \
python plot/cholesky_plotter.py \
    matrix-size-comparison \
    "${PLOT_DIR}" \
    "${RAW_CSVS[@]}"