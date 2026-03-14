#!/usr/bin/env bash

# Exit on the first error, on unset variables, and on failures inside pipelines.
set -euo pipefail

# Resolve the directory that contains this script.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Load shared project paths and executable locations.
source "${SCRIPT_DIR}/common_run_config.sh"

# Read the repeat count from the first argument, defaulting to 3 so the full
# all-method sweep stays within the 5-hour wall-clock cap on CSD3.
REPEATS="${1:-3}"

# Allow the plotting conda environment name to be overridden by the caller.
CONDA_ENV_NAME="${CONDA_ENV_NAME:-coursework-plot}"

# Store the project-level conda environment definition file.
ENV_FILE="${ROOT_DIR}/environment.yml"

# Drop the first argument so any remaining arguments can be treated as matrix sizes.
shift $(( $# >= 1 ? 1 : $# ))

# If the caller did not provide sizes, use a default sweep that is large enough to show trends.
if [ "$#" -eq 0 ]; then
    SIZES=(2000 4000 6000 8000 10000)
else
    SIZES=("$@")
fi

# Benchmark the currently implemented single-threaded methods.
OPTIMISATION_METHODS=(
    baseline
    lower_triangle
    upper_triangle
    contiguous_access
    cache_blocked_1
    cache_blocked_2
)

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

# Stop early with a clear error if conda is not installed or not on PATH.
if ! command -v conda >/dev/null 2>&1; then
    echo "Error: conda is required for plotting but was not found on PATH." >&2
    exit 1
fi

# Stop early with a clear error if the project environment file is missing.
if [ ! -f "${ENV_FILE}" ]; then
    echo "Error: missing conda environment file: ${ENV_FILE}" >&2
    exit 1
fi

# Load conda into this non-interactive shell.
source "$(conda info --base)/etc/profile.d/conda.sh"

# Create the plotting environment from environment.yml if it does not already exist.
if ! conda env list | awk 'NF && $1 !~ /^#/ { print $1 }' | grep -Fxq "${CONDA_ENV_NAME}"; then
    echo "==> Creating conda environment: ${CONDA_ENV_NAME}"
    conda env create --yes -n "${CONDA_ENV_NAME}" -f "${ENV_FILE}"
fi

# Activate the plotting environment before running the scaling driver and comparison plotter.
conda activate "${CONDA_ENV_NAME}"

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
python3 "${ROOT_DIR}/plot/plot_comparison_metrics.py" \
    "${COMPARISON_FIG_DIR}" \
    "${RAW_CSV}"

# Print the output locations for the report-ready plots.
echo "==> Raw CSV: ${RAW_CSV}"
echo "==> Plot directory: ${COMPARISON_FIG_DIR}"
