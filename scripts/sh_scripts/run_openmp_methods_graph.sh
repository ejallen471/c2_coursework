#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common_run_config.sh"

REPEATS="${1:-3}"
OPENMP_THREADS="${OPENMP_THREADS:-${SLURM_CPUS_PER_TASK:-8}}"

shift $(( $# >= 1 ? 1 : $# ))

if [ "$#" -eq 0 ]; then
    SIZES=(512 1024 2048 4096)
else
    SIZES=("$@")
fi

METHODS=(baseline openmp1 openmp2 openmp3 openmp4)
BUILD_DIR="${OPENMP_BUILD_DIR}"
RAW_CSV="${RESULTS_RAW_DIR}/openmp_methods_graph.csv"
PLOT_DIR="${RESULTS_FIG_DIR}/openmp_methods_graph"

mkdir -p "${RESULTS_RAW_DIR}" "${PLOT_DIR}"

echo "==> OpenMP matrix size comparison"
echo "==> Repeats: ${REPEATS}"
echo "==> Sizes: ${SIZES[*]}"
echo "==> Methods: ${METHODS[*]}"
echo "==> OpenMP threads: ${OPENMP_THREADS}"

ensure_release_executable_in_build "${BUILD_DIR}" "${RUN_CHOLESKY_EXEC_REL}" "${OPENMP_BUILD_SCRIPT}"

if ! command -v python >/dev/null 2>&1; then
    echo "Error: python is required for plotting but was not found on PATH." >&2
    exit 1
fi

TEMP_RAW_DIR="$(mktemp -d "${TMPDIR:-/tmp}/openmp_methods_graph.XXXXXX")"
trap 'rm -rf "${TEMP_RAW_DIR}"' EXIT

FIRST_CSV=1

for METHOD in "${METHODS[@]}"; do
    METHOD_RAW_CSV="${TEMP_RAW_DIR}/${METHOD}.csv"

    echo "==> Running ${METHOD}"

    if [ "${METHOD}" = "baseline" ]; then
        OMP_NUM_THREADS=1 "${BUILD_DIR}/${RUN_CHOLESKY_EXEC_REL}" \
            scaling \
            "${METHOD}" \
            "${REPEATS}" \
            "${METHOD_RAW_CSV}" \
            "${SIZES[@]}"
    else
        OMP_NUM_THREADS="${OPENMP_THREADS}" "${BUILD_DIR}/${RUN_CHOLESKY_EXEC_REL}" \
            scaling \
            "${METHOD}" \
            "${REPEATS}" \
            "${METHOD_RAW_CSV}" \
            "${SIZES[@]}"
    fi

    if [ "${FIRST_CSV}" -eq 1 ]; then
        cp "${METHOD_RAW_CSV}" "${RAW_CSV}"
        FIRST_CSV=0
    else
        tail -n +2 "${METHOD_RAW_CSV}" >> "${RAW_CSV}"
    fi
done

MPLCONFIGDIR=/tmp/mpl_perf_graph \
python "${ROOT_DIR}/plot/plot_comparison_metrics.py" \
    "${PLOT_DIR}" \
    "${RAW_CSV}"

echo "==> Raw CSV: ${RAW_CSV}"
echo "==> Plot directory: ${PLOT_DIR}"
