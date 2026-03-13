#!/usr/bin/env bash

# Exit on the first error, on unset variables, and on failures inside pipelines.
set -euo pipefail

# Resolve the directory that contains this script.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Load shared project paths and executable locations.
source "${SCRIPT_DIR}/common_run_config.sh"

# Use one shared build directory for all normal project runs.
BUILD_DIR="${SHARED_BUILD_DIR}"

# Ensure the build directory exists.
mkdir -p "${BUILD_DIR}"

# If environment modules are available, load the standard toolchain.
if type module >/dev/null 2>&1; then
    module purge
    module load cmake
    module load gcc
fi

# Print the build configuration so the user can see which directory is being populated.
echo "==> Shared project build"
echo "==> Root directory: ${ROOT_DIR}"
echo "==> Build directory: ${BUILD_DIR}"
echo "==> Build type: ${SHARED_BUILD_TYPE}"

# Configure a release build of the project in the shared build directory.
cmake -S "${ROOT_DIR}" -B "${BUILD_DIR}" -DCMAKE_BUILD_TYPE="${SHARED_BUILD_TYPE}"

# Build the executables used by the local and Slurm workflows.
cmake --build "${BUILD_DIR}" --target example_cholesky perf_time perf_fixed_size_comparison perf_block_size_sweep perf_scaling --parallel

# Print the key executables produced by the build.
echo "==> Built example executable: ${BUILD_DIR}/${EXAMPLE_EXEC_REL}"
echo "==> Built single-matrix executable: ${BUILD_DIR}/${PERF_TIME_EXEC_REL}"
echo "==> Built fixed-size comparison executable: ${BUILD_DIR}/${PERF_FIXED_SIZE_EXEC_REL}"
echo "==> Built block-size sweep executable: ${BUILD_DIR}/${PERF_BLOCK_SIZE_SWEEP_EXEC_REL}"
echo "==> Built scaling executable: ${BUILD_DIR}/${PERF_SCALING_EXEC_REL}"
