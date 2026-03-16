#!/usr/bin/env bash

# Exit on the first error, on unset variables, and on failures inside pipelines.
set -euo pipefail

# Resolve the directory that contains this script.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Load shared project paths and executable locations.
source "${SCRIPT_DIR}/common_run_config.sh"

print_usage() {
    cat <<'EOF'
Usage:
  bash scripts/sh_scripts/run_all_methods_graph.sh [repeats] [n1 n2 ...]
  bash scripts/sh_scripts/run_all_methods_graph.sh [repeats] --methods <method1> [method2 ...] --sizes <n1> [n2 ...]

Examples:
  bash scripts/sh_scripts/run_all_methods_graph.sh 3 2000 4000 6000
  bash scripts/sh_scripts/run_all_methods_graph.sh 3 --methods lower_triangle upper_triangle contiguous_access cache_blocked_1 cache_blocked_2 --sizes 256 384 512 768 1000 1500 2000
EOF
}

DEFAULT_REPEATS=3
DEFAULT_SIZES=(2000 4000 6000 8000 10000)
DEFAULT_METHODS=(
    baseline
    lower_triangle
    upper_triangle
    contiguous_access
    cache_blocked_1
    cache_blocked_2
)

REPEATS="${DEFAULT_REPEATS}"

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

OPTIMISATION_METHODS=("${DEFAULT_METHODS[@]}")
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
            OPTIMISATION_METHODS=()
            shift

            while [ "$#" -gt 0 ] && [ "$1" != "--sizes" ] && [ "$1" != "--methods" ]; do
                OPTIMISATION_METHODS+=("$1")
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

if [ "${#OPTIMISATION_METHODS[@]}" -eq 0 ]; then
    echo "Error: at least one optimisation method must be provided." >&2
    print_usage >&2
    exit 1
fi

if [ "${#SIZES[@]}" -eq 0 ]; then
    SIZES=("${DEFAULT_SIZES[@]}")
fi

# Use the shared prebuilt project directory.
BUILD_DIR="${SHARED_BUILD_DIR}"

# Store one combined raw CSV for the whole workflow.
RAW_CSV="${RESULTS_RAW_DIR}/all_methods_graph.csv"

# Store the combined comparison figures in one directory.
COMPARISON_FIG_DIR="${RESULTS_FIG_DIR}/all_methods_graph"

# Ensure the output directories exist.
mkdir -p "${RESULTS_RAW_DIR}" "${COMPARISON_FIG_DIR}"

# Print the requested run configuration.
echo "==> All-method matrix size comparison"
echo "==> Repeats: ${REPEATS}"
echo "==> Sizes: ${SIZES[*]}"
echo "==> Methods: ${OPTIMISATION_METHODS[*]}"

# Ensure the shared executable exists and was built with the optimised build type.
ensure_shared_release_executable "${RUN_CHOLESKY_EXEC_REL}"
PLOT_PYTHON="$(resolve_plot_python)"

# Build temporary per-method CSVs, merge them into one combined raw CSV, then remove the
# intermediates so the user is left with one CSV for the whole workflow.
TEMP_RAW_DIR="$(mktemp -d "${TMPDIR:-/tmp}/all_methods_graph.XXXXXX")"
trap 'rm -rf "${TEMP_RAW_DIR}"' EXIT

FIRST_CSV=1

# Run the scaling benchmark once per implementation.
for OPTIMISATION_METHOD in "${OPTIMISATION_METHODS[@]}"; do
    METHOD_RAW_CSV="${TEMP_RAW_DIR}/${OPTIMISATION_METHOD}.csv"

    echo "==> Running ${OPTIMISATION_METHOD}"

    "${BUILD_DIR}/${RUN_CHOLESKY_EXEC_REL}" \
        scaling \
        "${OPTIMISATION_METHOD}" \
        "${REPEATS}" \
        "${METHOD_RAW_CSV}" \
        "${SIZES[@]}"

    if [ "${FIRST_CSV}" -eq 1 ]; then
        cp "${METHOD_RAW_CSV}" "${RAW_CSV}"
        FIRST_CSV=0
    else
        tail -n +2 "${METHOD_RAW_CSV}" >> "${RAW_CSV}"
    fi
done

# Generate the combined comparison figures from the one merged raw CSV.
MPLCONFIGDIR=/tmp/mpl_perf_graph \
"${PLOT_PYTHON}" "${ROOT_DIR}/plot/plot_comparison_metrics.py" \
    "${COMPARISON_FIG_DIR}" \
    "${RAW_CSV}"

# Print the output locations for the report-ready plots.
echo "==> Raw CSV: ${RAW_CSV}"
echo "==> Plot directory: ${COMPARISON_FIG_DIR}"
