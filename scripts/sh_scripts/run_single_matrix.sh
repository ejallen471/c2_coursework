#!/usr/bin/env bash

# Exit on the first error, on unset variables, and on failures inside pipelines.
set -euo pipefail

# Figure out what directory this script is in
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Load shared project paths and executable locations from common_run_config.sh
source "${SCRIPT_DIR}/common_run_config.sh"

# Read the requested optimisation method from the first argument, defaulting to baseline.
# Options are baseline, lower_triangle_only, inline_mirror, loop_cleanup, access_pattern,
# cache_blocked, vectorisation, blocked_vectorised
OPTIMISATION_METHOD="${1:-baseline}"

# Read the requested matrix size from the second argument, defaulting to 2000.
MATRIX_SIZE="${2:-2000}"

# Use the shared prebuilt project directory.
BUILD_DIR="${SHARED_BUILD_DIR}"

# Print a short description of this run.
echo "==> Single matrix timing"

# Show which optimisation will be timed.
echo "==> Optimisation: ${OPTIMISATION_METHOD}"

# Show the matrix size that will be generated and factorised.
echo "==> Matrix size: ${MATRIX_SIZE}"

# Ensure the shared executable exists and was built with the optimised build type.
ensure_shared_release_executable "${PERF_TIME_EXEC_REL}"

# Run the timing executable with the requested optimisation method and matrix size.
"${BUILD_DIR}/${PERF_TIME_EXEC_REL}" "${OPTIMISATION_METHOD}" "${MATRIX_SIZE}"
