#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common_run_config.sh"

MATRIX_SIZE="${1:-2000}"
REPEATS="${2:-1}"
OPENMP_THREADS="${OPENMP_THREADS:-${SLURM_CPUS_PER_TASK:-8}}"
RAW_CSV="${3:-${RESULTS_RAW_DIR}/openmp_fixed_size_comparison_n${MATRIX_SIZE}.csv}"

BUILD_DIR="${OPENMP_BUILD_DIR}"
METHODS=(baseline openmp1 openmp2 openmp3 openmp4)

mkdir -p "$(dirname "${RAW_CSV}")"

echo "==> OpenMP fixed-size comparison"
echo "==> Matrix size: ${MATRIX_SIZE}"
echo "==> Repeats: ${REPEATS}"
echo "==> OpenMP threads: ${OPENMP_THREADS}"
echo "==> Methods: ${METHODS[*]}"
echo "==> Raw CSV: ${RAW_CSV}"

ensure_release_executable_in_build "${BUILD_DIR}" "${RUN_CHOLESKY_EXEC_REL}" "${OPENMP_BUILD_SCRIPT}"

TEMP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/openmp_fixed_size.XXXXXX")"
trap 'rm -rf "${TEMP_DIR}"' EXIT

declare -a BASELINE_ELAPSED

printf '%s\n' \
    "optimisation,n,repeat,elapsed_seconds,speedup_factor_vs_baseline,logdet_library,logdet_factor,relative_difference_percent" \
    > "${RAW_CSV}"

run_one() {
    local method="$1"
    local repeat="$2"
    local temp_csv="${TEMP_DIR}/${method}_${repeat}.csv"
    local row=""

    if [ "${method}" = "baseline" ]; then
        OMP_NUM_THREADS=1 "${BUILD_DIR}/${RUN_CHOLESKY_EXEC_REL}" \
            time \
            "${method}" \
            "${MATRIX_SIZE}" \
            "${temp_csv}" >/dev/null
    else
        OMP_NUM_THREADS="${OPENMP_THREADS}" "${BUILD_DIR}/${RUN_CHOLESKY_EXEC_REL}" \
            time \
            "${method}" \
            "${MATRIX_SIZE}" \
            "${temp_csv}" >/dev/null
    fi

    row="$(tail -n 1 "${temp_csv}")"
    printf '%s\n' "${row}"
}

for (( repeat=0; repeat<REPEATS; ++repeat )); do
    row="$(run_one baseline "${repeat}")"
    IFS=',' read -r optimisation n elapsed logdet_library logdet_factor relative_difference <<< "${row}"
    BASELINE_ELAPSED[repeat]="${elapsed}"
    printf '%s,%s,%s,%s,%s,%s,%s,%s\n' \
        "${optimisation}" \
        "${n}" \
        "${repeat}" \
        "${elapsed}" \
        "1" \
        "${logdet_library}" \
        "${logdet_factor}" \
        "${relative_difference}" >> "${RAW_CSV}"
done

for method in openmp1 openmp2 openmp3 openmp4; do
    for (( repeat=0; repeat<REPEATS; ++repeat )); do
        row="$(run_one "${method}" "${repeat}")"
        IFS=',' read -r optimisation n elapsed logdet_library logdet_factor relative_difference <<< "${row}"
        speedup="$(awk -v baseline="${BASELINE_ELAPSED[repeat]}" -v elapsed="${elapsed}" 'BEGIN { printf "%.16g", baseline / elapsed }')"
        printf '%s,%s,%s,%s,%s,%s,%s,%s\n' \
            "${optimisation}" \
            "${n}" \
            "${repeat}" \
            "${elapsed}" \
            "${speedup}" \
            "${logdet_library}" \
            "${logdet_factor}" \
            "${relative_difference}" >> "${RAW_CSV}"
    done
done

echo "raw_csv=\"${RAW_CSV}\""
