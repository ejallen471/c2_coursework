#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common_run_config.sh"

MATRIX_SIZE="${1:-5000}"
REPEATS="${2:-5}"

CONDA_ENV_NAME="${CONDA_ENV_NAME:-coursework-plot}"
ENV_FILE="${ROOT_DIR}/environment.yml"
BUILD_DIR="${SHARED_BUILD_DIR}"
WARMUP_OPTION="$(perf_warmup_cli_arg)"

shift $(( $# >= 2 ? 2 : $# ))

if [ "$#" -eq 0 ]; then
    BLOCK_SIZES=(16 24 32 40 48 64 96 128)
else
    BLOCK_SIZES=("$@")
fi

RAW_CSV="${RESULTS_RAW_DIR}/block_size_sweep_n${MATRIX_SIZE}.csv"
PLOT_DIR="${RESULTS_FIG_DIR}/block_size_sweep_n${MATRIX_SIZE}"

mkdir -p "${RESULTS_RAW_DIR}" "${RESULTS_FIG_DIR}"

echo "==> Block-size sweep"
echo "==> Matrix size: ${MATRIX_SIZE}"
echo "==> Repeats: ${REPEATS}"
echo "==> Block sizes: ${BLOCK_SIZES[*]}"

ensure_shared_release_executable "${PERF_BLOCK_SIZE_SWEEP_EXEC_REL}"

if ! command -v conda >/dev/null 2>&1; then
    echo "Error: conda is required for plotting but was not found on PATH." >&2
    exit 1
fi

if [ ! -f "${ENV_FILE}" ]; then
    echo "Error: missing conda environment file: ${ENV_FILE}" >&2
    exit 1
fi

source "$(conda info --base)/etc/profile.d/conda.sh"

if ! conda env list | awk 'NF && $1 !~ /^#/ { print $1 }' | grep -Fxq "${CONDA_ENV_NAME}"; then
    echo "==> Creating conda environment: ${CONDA_ENV_NAME}"
    conda env create --yes -n "${CONDA_ENV_NAME}" -f "${ENV_FILE}"
fi

conda activate "${CONDA_ENV_NAME}"

MPLCONFIGDIR=/tmp/mpl_perf_graph \
"${BUILD_DIR}/${PERF_BLOCK_SIZE_SWEEP_EXEC_REL}" \
    "${MATRIX_SIZE}" \
    "${REPEATS}" \
    "${RAW_CSV}" \
    "${PLOT_DIR}" \
    "${WARMUP_OPTION}" \
    "${BLOCK_SIZES[@]}"

echo "==> Raw CSV: ${RAW_CSV}"
echo "==> Plot directory: ${PLOT_DIR}"
