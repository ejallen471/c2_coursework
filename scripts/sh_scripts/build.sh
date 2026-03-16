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

# Initialise the module command in non-interactive batch shells when available.
if [ -f /etc/profile.d/modules.sh ]; then
    . /etc/profile.d/modules.sh
fi

# If environment modules are available, load the standard toolchain.
if type module >/dev/null 2>&1; then
    module purge

    if [ "${SLURM_JOB_PARTITION:-${SLURM_PARTITION:-}}" = "icelake" ]; then
        module load rhel8/default-icl
    fi

    module load cmake
    module load gcc
fi

# Keep the build parallelism inside the CPU allocation when running under Slurm.
if [ -n "${SLURM_CPUS_PER_TASK:-}" ]; then
    export CMAKE_BUILD_PARALLEL_LEVEL="${CMAKE_BUILD_PARALLEL_LEVEL:-${SLURM_CPUS_PER_TASK}}"
fi

# Print the build configuration so the user can see which directory is being populated.
echo "==> Shared project build"
echo "==> Root directory: ${ROOT_DIR}"
echo "==> Build directory: ${BUILD_DIR}"
echo "==> Build type: ${SHARED_BUILD_TYPE}"

# Configure a release build of the project in the shared build directory.
cmake -S "${ROOT_DIR}" -B "${BUILD_DIR}" -DCMAKE_BUILD_TYPE="${SHARED_BUILD_TYPE}"

# Build the executables used by the local and Slurm workflows.
cmake --build "${BUILD_DIR}" --target run_cholesky compare_matrix_generators --parallel

# Print the key executable produced by the build.
echo "==> Built benchmark executable: ${BUILD_DIR}/${RUN_CHOLESKY_EXEC_REL}"
