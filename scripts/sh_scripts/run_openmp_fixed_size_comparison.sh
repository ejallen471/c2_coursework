#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common_run_config.sh"

MATRIX_SIZE="${1:-5000}"
REPEATS="${2:-1}"
OPENMP_THREADS="${OPENMP_THREADS:-${SLURM_CPUS_PER_TASK:-8}}"
RAW_CSV="${RESULTS_RAW_DIR}/openmp_fixed_size_comparison_n${MATRIX_SIZE}.csv"

BUILD_DIR="${OPENMP_BUILD_DIR}"
METHODS=(baseline openmp1 openmp2 openmp3 openmp4)

shift $(( $# >= 2 ? 2 : $# ))

canonical_method_name() {
    case "$1" in
        baseline)
            printf '%s\n' "baseline"
            ;;
        openmp1|openmp2|openmp3|openmp4)
            printf '%s\n' "$1"
            ;;
        *)
            return 1
            ;;
    esac
}

if [ "$#" -gt 0 ] && [ "$1" != "--methods" ]; then
    RAW_CSV="$1"
    shift
fi

if [ "$#" -gt 0 ]; then
    if [ "$1" != "--methods" ]; then
        echo "Error: expected '--methods' before the OpenMP method list." >&2
        exit 1
    fi

    shift

    if [ "$#" -eq 0 ]; then
        echo "Error: '--methods' requires at least one method." >&2
        exit 1
    fi

    METHODS=()

    for method in "$@"; do
        method_already_selected=0
        canonical_method="$(canonical_method_name "${method}")" || {
            echo "Error: unknown OpenMP fixed-size method '${method}'." >&2
            exit 1
        }

        if [ "${#METHODS[@]}" -gt 0 ]; then
            for existing_method in "${METHODS[@]}"; do
                if [ "${existing_method}" = "${canonical_method}" ]; then
                    method_already_selected=1
                    break
                fi
            done
        fi

        if [ "${method_already_selected}" -eq 0 ]; then
            METHODS+=("${canonical_method}")
        fi
    done
fi

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
HAVE_BASELINE=0

for method in "${METHODS[@]}"; do
    if [ "${method}" = "baseline" ]; then
        HAVE_BASELINE=1
        break
    fi
done

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
            "${temp_csv}" >/dev/null || return 1
    else
        OMP_NUM_THREADS="${OPENMP_THREADS}" "${BUILD_DIR}/${RUN_CHOLESKY_EXEC_REL}" \
            time \
            "${method}" \
            "${MATRIX_SIZE}" \
            "${temp_csv}" >/dev/null || return 1
    fi

    if [ ! -s "${temp_csv}" ]; then
        echo "Error: benchmark output CSV was not created for ${method} repeat ${repeat}" >&2
        return 1
    fi

    row="$(tail -n 1 "${temp_csv}")" || return 1
    if [ -z "${row}" ]; then
        echo "Error: benchmark output CSV is empty for ${method} repeat ${repeat}" >&2
        return 1
    fi

    printf '%s\n' "${row}"
}

if [ "${HAVE_BASELINE}" -eq 1 ]; then
    for (( repeat=0; repeat<REPEATS; ++repeat )); do
        if ! row="$(run_one baseline "${repeat}")"; then
            exit 1
        fi
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
fi

for method in "${METHODS[@]}"; do
    if [ "${method}" = "baseline" ]; then
        continue
    fi

    for (( repeat=0; repeat<REPEATS; ++repeat )); do
        if ! row="$(run_one "${method}" "${repeat}")"; then
            exit 1
        fi
        IFS=',' read -r optimisation n elapsed logdet_library logdet_factor relative_difference <<< "${row}"

        if [ "${HAVE_BASELINE}" -eq 1 ]; then
            speedup="$(awk -v baseline="${BASELINE_ELAPSED[repeat]}" -v elapsed="${elapsed}" 'BEGIN { printf "%.16g", baseline / elapsed }')"
        else
            speedup="nan"
        fi

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
