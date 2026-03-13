#!/bin/bash
#SBATCH -J chol-compiler
#SBATCH -A MPHIL-DIS-SL2-CPU
#SBATCH -o logs/chol_compiler_%j.out
#SBATCH -e logs/chol_compiler_%j.err
#SBATCH -p icelake
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --time=02:00:00

set -euo pipefail

# Run this script from the repository root with:
# sbatch scripts/slurm_scripts/run_compiler_optimisations.slurm <matrix_size> [O1 O2 O3 Ofast]

# Resolve the repository root from the directory where sbatch was launched.
ROOT_DIR="${SLURM_SUBMIT_DIR}"

# Read the matrix size from the first argument, defaulting to 5000.
MATRIX_SIZE="${1:-5000}"

# Drop the first argument so any remaining arguments can be treated as compiler flags.
shift $(( $# >= 1 ? 1 : $# ))

# If the caller did not provide a compiler-flag list, benchmark a standard optimisation sweep.
if [ "$#" -eq 0 ]; then
    COMPILER_FLAGS=(O1 O2 O3 Ofast)
else
    COMPILER_FLAGS=("$@")
fi

# Store logs and raw CSV output under the repository root.
LOG_DIR="${ROOT_DIR}/logs"
RESULTS_RAW_DIR="${ROOT_DIR}/results/raw"

# Relative path to the single-matrix timing executable inside each build directory.
PERF_TIME_EXEC_REL="performance_tests/perf_time"

# Write one raw CSV file per matrix size.
RAW_CSV="${RESULTS_RAW_DIR}/baseline_compiler_optimisations_n${MATRIX_SIZE}.csv"

# Ensure the output directories exist.
mkdir -p "${LOG_DIR}" "${RESULTS_RAW_DIR}"

# Load the standard CSD3 toolchain modules needed to run the prebuilt executables.
module purge
module load gcc

# Print the requested run configuration.
echo "==> Slurm compiler-optimisation benchmark"
echo "==> Root directory: ${ROOT_DIR}"
echo "==> Matrix size: ${MATRIX_SIZE}"
echo "==> Compiler flags: ${COMPILER_FLAGS[*]}"
echo "==> Output CSV: ${RAW_CSV}"

# Start a fresh CSV file for this benchmark sweep.
printf "compiler_flag,n,elapsed_seconds\n" > "${RAW_CSV}"

# Run the baseline implementation once for each prebuilt compiler-flag variant.
for FLAG_NAME in "${COMPILER_FLAGS[@]}"; do
    BUILD_DIR="${ROOT_DIR}/build_compiler_${FLAG_NAME}"
    EXECUTABLE="${BUILD_DIR}/${PERF_TIME_EXEC_REL}"

    # Stop early with a clear error if the required executable has not been built yet.
    if [ ! -x "${EXECUTABLE}" ]; then
        echo "Error: missing executable ${EXECUTABLE}" >&2
        echo "Build the compiler-flag variants first with scripts/sh_scripts/build_compiler_optimisations.sh" >&2
        exit 1
    fi

    # Show which compiler-flag variant is being run.
    echo "==> Running baseline built with -${FLAG_NAME}"

    # Capture the timing line so the elapsed time can be extracted into a CSV summary.
    OUTPUT="$("${EXECUTABLE}" baseline "${MATRIX_SIZE}")"

    # Print the full timing output to the job log for traceability.
    echo "${OUTPUT}"

    # Extract the elapsed time from the perf_time output format.
    ELAPSED_SECONDS="$(printf '%s\n' "${OUTPUT}" | awk '{for (i = 1; i <= NF; ++i) if ($i ~ /^elapsed_seconds=/) {split($i, a, "="); print a[2]}}')"

    # Append the compiler flag, matrix size, and elapsed time to the raw CSV.
    printf "%s,%s,%s\n" "${FLAG_NAME}" "${MATRIX_SIZE}" "${ELAPSED_SECONDS}" >> "${RAW_CSV}"
done

# Print the location of the generated CSV summary.
echo "==> Raw CSV: ${RAW_CSV}"
