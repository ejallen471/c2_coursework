#!/bin/bash
#SBATCH -J chol-fixed
#SBATCH -A MPHIL-DIS-SL2-CPU
#SBATCH -o logs/chol_fixed_%j.out
#SBATCH -e logs/chol_fixed_%j.err
#SBATCH -p icelake
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --time=01:00:00

set -euo pipefail

# Run this script from the repository root with:
# sbatch scripts/slurm_scripts/run_fixed_size_comparison.slurm <matrix_size>

# Resolve the repository root from the directory where sbatch was launched.
ROOT_DIR="${SLURM_SUBMIT_DIR}"

# Read the matrix size from the first argument, defaulting to 5000.
MATRIX_SIZE="${1:-5000}"

# Read the repeat count from the second argument, defaulting to 10.
REPEATS="${2:-10}"
PERF_WARMUP="${PERF_WARMUP:-1}"

# Use the shared prebuilt build directory.
BUILD_DIR="${ROOT_DIR}/build"

# Store logs and raw CSV output under the repository root.
LOG_DIR="${ROOT_DIR}/logs"
RESULTS_RAW_DIR="${ROOT_DIR}/results/raw"
RAW_CSV="${RESULTS_RAW_DIR}/fixed_size_comparison_n${MATRIX_SIZE}_raw.csv"

# Relative path to the fixed-size comparison executable inside the build directory.
PERF_FIXED_SIZE_EXEC_REL="performance_tests/perf_fixed_size_comparison"

# Ensure the output directories exist.
mkdir -p "${LOG_DIR}" "${RESULTS_RAW_DIR}"

# Load the standard CSD3 runtime module environment.
module purge
module load gcc

# Stop early with a clear error if the prebuilt executable is missing.
if [ ! -x "${BUILD_DIR}/${PERF_FIXED_SIZE_EXEC_REL}" ]; then
    echo "Error: missing executable ${BUILD_DIR}/${PERF_FIXED_SIZE_EXEC_REL}" >&2
    echo "Build the project first with scripts/sh_scripts/build.sh" >&2
    exit 1
fi

# Print the requested run configuration.
echo "==> Slurm fixed-size comparison"
echo "==> Matrix size: ${MATRIX_SIZE}"
echo "==> Repeats: ${REPEATS}"

case "${PERF_WARMUP}" in
    1|true|TRUE|on|ON|yes|YES)
        WARMUP_OPTION="--warmup"
        ;;
    0|false|FALSE|off|OFF|no|NO)
        WARMUP_OPTION="--no-warmup"
        ;;
    *)
        echo "Error: PERF_WARMUP must be one of 0/1, true/false, on/off, yes/no" >&2
        exit 1
        ;;
esac

# Run the fixed-size comparison executable so the raw CSV is produced in C++.
"${BUILD_DIR}/${PERF_FIXED_SIZE_EXEC_REL}" "${MATRIX_SIZE}" "${REPEATS}" "${RAW_CSV}" "${WARMUP_OPTION}" "${@:3}"

# Print the location of the generated CSV output.
echo "==> Raw CSV: ${RAW_CSV}"
