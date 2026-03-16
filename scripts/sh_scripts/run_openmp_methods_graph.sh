#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common_run_config.sh"

print_usage() {
    cat <<'EOF'
Usage:
  bash scripts/sh_scripts/run_openmp_methods_graph.sh [repeats] [n1 n2 ...]
  bash scripts/sh_scripts/run_openmp_methods_graph.sh [repeats] --methods <method1> [method2 ...] --sizes <n1> [n2 ...]

Examples:
  bash scripts/sh_scripts/run_openmp_methods_graph.sh 3 512 1024 2000 4096
  bash scripts/sh_scripts/run_openmp_methods_graph.sh 3 --methods openmp1 openmp2 openmp3 openmp4 --sizes 256 384 512 768 1000 1500 2000 3000 4000
EOF
}

DEFAULT_REPEATS=3
DEFAULT_SIZES=(512 1024 2048 4096)
DEFAULT_METHODS=(baseline openmp1 openmp2 openmp3 openmp4)

REPEATS="${DEFAULT_REPEATS}"
OPENMP_THREADS="${OPENMP_THREADS:-${SLURM_CPUS_PER_TASK:-8}}"

if [ "$#" -gt 0 ]; then
    case "$1" in
        --help|-h)
            print_usage
            exit 0
            ;;
        --methods|--sizes)
            ;;
        *)
            REPEATS="$1"
            shift
            ;;
    esac
fi

METHODS=("${DEFAULT_METHODS[@]}")
SIZES=()
USED_METHOD_FLAG=0
USED_SIZE_FLAG=0

while [ "$#" -gt 0 ]; do
    case "$1" in
        --help|-h)
            print_usage
            exit 0
            ;;
        --methods)
            USED_METHOD_FLAG=1
            METHODS=()
            shift

            while [ "$#" -gt 0 ] && [ "$1" != "--sizes" ] && [ "$1" != "--methods" ]; do
                METHODS+=("$1")
                shift
            done
            ;;
        --sizes)
            USED_SIZE_FLAG=1
            SIZES=()
            shift

            while [ "$#" -gt 0 ] && [ "$1" != "--methods" ] && [ "$1" != "--sizes" ]; do
                SIZES+=("$1")
                shift
            done
            ;;
        *)
            if [ "${USED_METHOD_FLAG}" -eq 0 ] && [ "${USED_SIZE_FLAG}" -eq 0 ]; then
                SIZES+=("$1")
                shift
            else
                echo "Error: unexpected argument '$1'" >&2
                print_usage >&2
                exit 1
            fi
            ;;
    esac
done

if [ "${#METHODS[@]}" -eq 0 ]; then
    echo "Error: at least one OpenMP optimisation method must be provided." >&2
    print_usage >&2
    exit 1
fi

if [ "${#SIZES[@]}" -eq 0 ]; then
    SIZES=("${DEFAULT_SIZES[@]}")
fi

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
