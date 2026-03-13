#!/usr/bin/env bash

# Exit on the first error, on unset variables, and on failures inside pipelines.
set -euo pipefail

# Resolve the directory that contains this script.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Load shared project paths and executable locations.
source "${SCRIPT_DIR}/common_run_config.sh"

# Read the requested matrix size from the first argument, defaulting to 5000.
MATRIX_SIZE="${1:-1000}"

# Read the repeat count from the second argument, defaulting to 10.
REPEATS="${2:-1}"

# Use the shared prebuilt project directory.
BUILD_DIR="${SHARED_BUILD_DIR}"
WARMUP_OPTION="$(perf_warmup_cli_arg)"

# Store the raw per-repeat CSV under the raw-results directory.
RAW_CSV="${RESULTS_RAW_DIR}/fixed_size_comparison_n${MATRIX_SIZE}_raw.csv"

# Ensure the output directory exists.
mkdir -p "${RESULTS_RAW_DIR}"

# Print the requested run configuration.
echo "==> Fixed-size comparison"
echo "==> Matrix size: ${MATRIX_SIZE}"
echo "==> Repeats: ${REPEATS}"

# Ensure the shared executable exists and was built with the optimised build type.
ensure_shared_release_executable "${PERF_FIXED_SIZE_EXEC_REL}"

# Run the fixed-size comparison executable so the raw CSV is produced in C++.
"${BUILD_DIR}/${PERF_FIXED_SIZE_EXEC_REL}" "${MATRIX_SIZE}" "${REPEATS}" "${RAW_CSV}" "${WARMUP_OPTION}" "${@:3}"

# Print the location of the generated CSV output.
echo "==> Raw CSV: ${RAW_CSV}"
