#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common_run_config.sh"

BUILD_DIR="${OPENMP_BUILD_DIR}"

mkdir -p "${BUILD_DIR}"

if type module >/dev/null 2>&1; then
    module purge
    module load cmake
    module load gcc
fi

echo "==> OpenMP project build"
echo "==> Root directory: ${ROOT_DIR}"
echo "==> Build directory: ${BUILD_DIR}"
echo "==> Build type: ${SHARED_BUILD_TYPE}"

if [ "$(uname -s)" = "Darwin" ]; then
    if ! command -v brew >/dev/null 2>&1; then
        echo "Error: Homebrew is required on macOS to build with OpenMP." >&2
        exit 1
    fi

    LLVM_PREFIX="$(brew --prefix llvm)"
    LIBOMP_PREFIX="$(brew --prefix libomp)"

    cmake -S "${ROOT_DIR}" -B "${BUILD_DIR}" \
        -DCMAKE_BUILD_TYPE="${SHARED_BUILD_TYPE}" \
        -DCMAKE_C_COMPILER="${LLVM_PREFIX}/bin/clang" \
        -DCMAKE_CXX_COMPILER="${LLVM_PREFIX}/bin/clang++" \
        -DCMAKE_PREFIX_PATH="${LLVM_PREFIX};${LIBOMP_PREFIX}"
else
    cmake -S "${ROOT_DIR}" -B "${BUILD_DIR}" -DCMAKE_BUILD_TYPE="${SHARED_BUILD_TYPE}"
fi

cmake --build "${BUILD_DIR}" --target run_cholesky --parallel

echo "==> Built OpenMP benchmark executable: ${BUILD_DIR}/${RUN_CHOLESKY_EXEC_REL}"
