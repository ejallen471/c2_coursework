#!/usr/bin/env bash

# Exit on the first error, on unset variables, and on failures inside pipelines.
set -euo pipefail

# Resolve the directory that contains this script.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Load shared project paths and executable locations.
source "${SCRIPT_DIR}/common_run_config.sh"

# Build all optimised compiler-flag variants unless the caller overrides the list.
if [ "$#" -eq 0 ]; then
    COMPILER_FLAGS=(O1 O2 O3 Ofast)
else
    COMPILER_FLAGS=("$@")
fi

# Load the standard CSD3 toolchain modules needed to configure and build the project.
module purge
module load cmake
module load gcc

# Print the build configuration before starting.
echo "==> Compiler-optimisation build"
echo "==> Root directory: ${ROOT_DIR}"
echo "==> Variants: ${COMPILER_FLAGS[*]}"

# Build one separate tree per compiler flag so results do not interfere with each other.
for FLAG_NAME in "${COMPILER_FLAGS[@]}"; do
    BUILD_DIR="${ROOT_DIR}/build_compiler_${FLAG_NAME}"
    FLAG_VALUE="-${FLAG_NAME}"

    # Ensure the build directory exists before configuring it.
    mkdir -p "${BUILD_DIR}"

    # Show which build variant is being configured.
    echo "==> Configuring ${FLAG_VALUE} in ${BUILD_DIR}"

    # Configure a release build that overrides the default release optimisation level.
    cmake -S "${ROOT_DIR}" \
        -B "${BUILD_DIR}" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CXX_FLAGS_RELEASE="${FLAG_VALUE} -DNDEBUG"

    # Build only the single-run timing executable needed for the compiler-flag study.
    cmake --build "${BUILD_DIR}" --target perf_time --parallel

    # Print the path to the produced executable for this compiler-flag variant.
    echo "==> Built ${BUILD_DIR}/${PERF_TIME_EXEC_REL}"
done
