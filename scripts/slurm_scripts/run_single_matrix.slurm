#!/bin/bash
#SBATCH -J chol-single
#SBATCH -A MPHIL-DIS-SL2-CPU
#SBATCH -o logs/chol_single_%j.out
#SBATCH -e logs/chol_single_%j.err
#SBATCH -p icelake
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --time=00:30:00

set -euo pipefail

# Run this script from the repository root with:
# sbatch scripts/slurm_scripts/run_single_matrix.slurm <optimisation> <matrix_size>

# Resolve the repository root from the directory where sbatch was launched.
ROOT_DIR="${SLURM_SUBMIT_DIR}"

# Read the optimisation method from the first argument, defaulting to baseline.
# Options are: baseline, lower_triangle_only, inline_mirror, loop_cleanup,
# access_pattern, cache_blocked, vectorisation, blocked_vectorised
OPTIMISATION_METHOD="${1:-baseline}"

# Read the matrix size from the second argument, defaulting to 5000.
DEFAULT_MATRIX_SIZE=5000
MATRIX_SIZE="${2:-${DEFAULT_MATRIX_SIZE}}"

# Use the shared build directory created ahead of time by the build script.
BUILD_DIR="${ROOT_DIR}/build"

# Store logs under the repository root.
LOG_DIR="${ROOT_DIR}/logs"

# Relative path to the single-matrix timing executable inside the build directory.
PERF_TIME_EXEC_REL="performance_tests/perf_time"

# Ensure the log directory exists.
mkdir -p "${LOG_DIR}"

# Load the standard CSD3 toolchain modules needed to run the prebuilt executable.
module purge
module load gcc

# Print the requested run configuration.
echo "==> Slurm single matrix timing"
echo "==> Root directory: ${ROOT_DIR}"
echo "==> Build directory: ${BUILD_DIR}"
echo "==> Optimisation: ${OPTIMISATION_METHOD}"
echo "==> Matrix size: ${MATRIX_SIZE}"

# Stop early with a clear error if the prebuilt executable is missing.
if [ ! -x "${BUILD_DIR}/${PERF_TIME_EXEC_REL}" ]; then
    echo "Error: missing executable ${BUILD_DIR}/${PERF_TIME_EXEC_REL}" >&2
    echo "Build the project first with scripts/sh_scripts/build.sh" >&2
    exit 1
fi

# Run the timing executable with the requested optimisation and matrix size.
"${BUILD_DIR}/${PERF_TIME_EXEC_REL}" "${OPTIMISATION_METHOD}" "${MATRIX_SIZE}"
