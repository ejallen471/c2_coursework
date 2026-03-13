#!/usr/bin/env bash

# Exit on the first error, on unset variables, and on failures inside pipelines.
set -euo pipefail

# Figure out what directory this script is in
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Load shared project paths and executable locations from common_run_config.sh
source "${SCRIPT_DIR}/common_run_config.sh"

# Use the shared prebuilt project directory.
BUILD_DIR="${SHARED_BUILD_DIR}"

# Print a short status message so the user knows what this script is doing.
echo "==> Running example/example_cholesky.cpp"

# Ensure the shared executable exists and was built with the optimised build type.
ensure_shared_release_executable "${EXAMPLE_EXEC_REL}"

# Run the built example executable.
"${BUILD_DIR}/${EXAMPLE_EXEC_REL}"
