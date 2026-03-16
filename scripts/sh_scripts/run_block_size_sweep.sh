#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common_run_config.sh"

OPTIMISATION_METHOD="${1:-cache_blocked_1}"
MATRIX_SIZE="${2:-5000}"
REPEATS="${3:-5}"
BUILD_DIR="${SHARED_BUILD_DIR}"

case "${OPTIMISATION_METHOD}" in
    cache_blocked_1|cache_blocked1|cache_blocked_2|cache_blocked2)
        ;;
    *)
    OPTIMISATION_METHOD="cache_blocked_1"
    MATRIX_SIZE="${1:-5000}"
    REPEATS="${2:-5}"
    shift $(( $# >= 2 ? 2 : $# ))
        ;;
esac

case "${OPTIMISATION_METHOD}" in
    cache_blocked1)
        OPTIMISATION_METHOD="cache_blocked_1"
        ;;
    cache_blocked2)
        OPTIMISATION_METHOD="cache_blocked_2"
        ;;
esac

if [[ "${1:-cache_blocked_1}" == "cache_blocked_1" || "${1:-cache_blocked_1}" == "cache_blocked1" || "${1:-cache_blocked_1}" == "cache_blocked_2" || "${1:-cache_blocked_1}" == "cache_blocked2" ]]; then
    shift $(( $# >= 3 ? 3 : $# ))
fi

if [ "$#" -eq 0 ]; then
    BLOCK_SIZES=(16 24 32 40 48 64 96 128)
else
    BLOCK_SIZES=("$@")
fi

RAW_CSV="${RESULTS_RAW_DIR}/${OPTIMISATION_METHOD}_block_size_sweep_n${MATRIX_SIZE}.csv"
PLOT_DIR="${RESULTS_FIG_DIR}/${OPTIMISATION_METHOD}_block_size_sweep_n${MATRIX_SIZE}"

mkdir -p "${RESULTS_RAW_DIR}" "${RESULTS_FIG_DIR}"

echo "==> Block-size sweep"
echo "==> Optimisation: ${OPTIMISATION_METHOD}"
echo "==> Matrix size: ${MATRIX_SIZE}"
echo "==> Repeats: ${REPEATS}"
echo "==> Block sizes: ${BLOCK_SIZES[*]}"

ensure_shared_release_executable "${RUN_CHOLESKY_EXEC_REL}"
PLOT_PYTHON="$(resolve_plot_python)"

# Run the block-size sweep first so the raw CSV exists before plotting.
"${BUILD_DIR}/${RUN_CHOLESKY_EXEC_REL}" \
    block-size-sweep \
    "${OPTIMISATION_METHOD}" \
    "${MATRIX_SIZE}" \
    "${REPEATS}" \
    "${RAW_CSV}" \
    "${BLOCK_SIZES[@]}"

MPLCONFIGDIR=/tmp/mpl_perf_graph \
"${PLOT_PYTHON}" "${ROOT_DIR}/plot/plot_block_size_metrics.py" \
    "${RAW_CSV}" \
    "${PLOT_DIR}"

echo "==> Raw CSV: ${RAW_CSV}"
echo "==> Plot directory: ${PLOT_DIR}"
