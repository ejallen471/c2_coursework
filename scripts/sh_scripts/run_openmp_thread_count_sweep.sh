#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common_run_config.sh"

MATRIX_SIZE="${1:-2000}"
REPEATS="${2:-3}"

shift $(( $# >= 2 ? 2 : $# ))

if [ "$#" -eq 0 ]; then
    THREAD_COUNTS=(1 2 4 8)
else
    THREAD_COUNTS=("$@")
fi

METHODS=(baseline openmp1 openmp2 openmp3 openmp4)
BUILD_DIR="${OPENMP_BUILD_DIR}"
RAW_CSV="${RESULTS_RAW_DIR}/openmp_thread_count_sweep_n${MATRIX_SIZE}.csv"
PLOT_DIR="${RESULTS_FIG_DIR}/openmp_thread_count_sweep_n${MATRIX_SIZE}"

mkdir -p "${RESULTS_RAW_DIR}" "${PLOT_DIR}"

echo "==> OpenMP thread-count sweep"
echo "==> Matrix size: ${MATRIX_SIZE}"
echo "==> Repeats: ${REPEATS}"
echo "==> Thread counts: ${THREAD_COUNTS[*]}"
echo "==> Methods: ${METHODS[*]}"

ensure_release_executable_in_build "${BUILD_DIR}" "${RUN_CHOLESKY_EXEC_REL}" "${OPENMP_BUILD_SCRIPT}"

if ! command -v python >/dev/null 2>&1; then
    echo "Error: python is required for plotting but was not found on PATH." >&2
    exit 1
fi

TEMP_RAW_DIR="$(mktemp -d "${TMPDIR:-/tmp}/openmp_thread_count_sweep.XXXXXX")"
trap 'rm -rf "${TEMP_RAW_DIR}"' EXIT

echo "tag,n,threads,repeat,elapsed_seconds,logdet,time_over_n3" > "${RAW_CSV}"

BASELINE_RAW_CSV="${TEMP_RAW_DIR}/baseline.csv"
OMP_NUM_THREADS=1 "${BUILD_DIR}/${RUN_CHOLESKY_EXEC_REL}" \
    scaling \
    baseline \
    "${REPEATS}" \
    "${BASELINE_RAW_CSV}" \
    "${MATRIX_SIZE}"

tail -n +2 "${BASELINE_RAW_CSV}" | \
    awk -F, 'BEGIN{OFS=","} {print $1,$2,1,$3,$4,$5,$6}' >> "${RAW_CSV}"

for METHOD in openmp1 openmp2 openmp3 openmp4; do
    for THREADS in "${THREAD_COUNTS[@]}"; do
        METHOD_RAW_CSV="${TEMP_RAW_DIR}/${METHOD}_${THREADS}.csv"

        echo "==> Running ${METHOD} with ${THREADS} threads"

        OMP_NUM_THREADS="${THREADS}" "${BUILD_DIR}/${RUN_CHOLESKY_EXEC_REL}" \
            scaling \
            "${METHOD}" \
            "${REPEATS}" \
            "${METHOD_RAW_CSV}" \
            "${MATRIX_SIZE}"

        tail -n +2 "${METHOD_RAW_CSV}" | \
            awk -F, -v threads="${THREADS}" 'BEGIN{OFS=","} {print $1,$2,threads,$3,$4,$5,$6}' >> "${RAW_CSV}"
    done
done

echo "==> Raw CSV: ${RAW_CSV}"

MPLCONFIGDIR=/tmp/mpl_perf_graph \
python "${ROOT_DIR}/plot/plot_openmp_thread_count_metrics.py" \
    "${RAW_CSV}" \
    "${PLOT_DIR}"

echo "==> Plot directory: ${PLOT_DIR}"
