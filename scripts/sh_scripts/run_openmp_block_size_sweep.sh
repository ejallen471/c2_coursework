#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common_run_config.sh"

OPTIMISATION_METHOD="${1:-openmp3}"
MATRIX_SIZE="${2:-5000}"
REPEATS="${3:-5}"
OPENMP_THREADS="${OPENMP_THREADS:-${SLURM_CPUS_PER_TASK:-8}}"
BUILD_DIR="${OPENMP_BUILD_DIR}"

case "${OPTIMISATION_METHOD}" in
    openmp3|openmp_3|openmp4|openmp_4)
        ;;
    *)
        OPTIMISATION_METHOD="openmp3"
        MATRIX_SIZE="${1:-5000}"
        REPEATS="${2:-5}"
        shift $(( $# >= 2 ? 2 : $# ))
        ;;
esac

case "${OPTIMISATION_METHOD}" in
    openmp_3)
        OPTIMISATION_METHOD="openmp3"
        ;;
    openmp_4)
        OPTIMISATION_METHOD="openmp4"
        ;;
esac

if [[ "${1:-openmp3}" == "openmp3" || "${1:-openmp3}" == "openmp_3" || \
      "${1:-openmp3}" == "openmp4" || "${1:-openmp3}" == "openmp_4" ]]; then
    shift $(( $# >= 3 ? 3 : $# ))
fi

if [ "$#" -eq 0 ]; then
    BLOCK_SIZES=(8 12 16 24 32 40 48 64 80 96 128)
else
    BLOCK_SIZES=("$@")
fi

RAW_CSV="${RESULTS_RAW_DIR}/${OPTIMISATION_METHOD}_block_size_sweep_n${MATRIX_SIZE}.csv"
PLOT_DIR="${RESULTS_FIG_DIR}/${OPTIMISATION_METHOD}_block_size_sweep_n${MATRIX_SIZE}"

mkdir -p "${RESULTS_RAW_DIR}" "${RESULTS_FIG_DIR}"

echo "==> OpenMP block-size sweep"
echo "==> Optimisation: ${OPTIMISATION_METHOD}"
echo "==> Matrix size: ${MATRIX_SIZE}"
echo "==> Repeats: ${REPEATS}"
echo "==> OpenMP threads: ${OPENMP_THREADS}"
echo "==> Block sizes: ${BLOCK_SIZES[*]}"

ensure_release_executable_in_build "${BUILD_DIR}" "${RUN_CHOLESKY_EXEC_REL}" "${OPENMP_BUILD_SCRIPT}"
PLOT_PYTHON="$(resolve_plot_python)"

OMP_NUM_THREADS="${OPENMP_THREADS}" "${BUILD_DIR}/${RUN_CHOLESKY_EXEC_REL}" \
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
