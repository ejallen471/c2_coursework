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
    lower_triangle_only
    inline_mirror
    loop_cleanup
    access_pattern
    cache_blocked
    vectorisation
    blocked_vectorised
)

# Use the shared prebuilt project directory.
BUILD_DIR="${SHARED_BUILD_DIR}"
WARMUP_OPTION="$(perf_warmup_cli_arg)"

# Store all-method raw CSV files in a dedicated subdirectory.
RAW_COMPARISON_DIR="${RESULTS_RAW_DIR}/all_methods_graph"

# Store per-method and combined figures in a dedicated comparison directory.
COMPARISON_FIG_DIR="${RESULTS_FIG_DIR}/all_methods_graph"

# Ensure the output directories exist.
mkdir -p "${RAW_COMPARISON_DIR}" "${COMPARISON_FIG_DIR}"

# Print the requested run configuration.
echo "==> All-method matrix size comparison"
echo "==> Repeats: ${REPEATS}"
echo "==> Sizes: ${SIZES[*]}"
echo "==> Methods: ${OPTIMISATION_METHODS[*]}"

# Ensure the shared executable exists and was built with the optimised build type.
ensure_shared_release_executable "${PERF_SCALING_EXEC_REL}"

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

# Collect the generated raw CSV files so the combined comparison plotter can consume them.
RAW_CSVS=()

# Run the scaling benchmark once per implementation.
for OPTIMISATION_METHOD in "${OPTIMISATION_METHODS[@]}"; do
    RAW_CSV="${RAW_COMPARISON_DIR}/${OPTIMISATION_METHOD}.csv"
    PLOT_DIR="${COMPARISON_FIG_DIR}/${OPTIMISATION_METHOD}"

    echo "==> Running ${OPTIMISATION_METHOD}"

    MPLCONFIGDIR=/tmp/mpl_perf_graph \
    "${BUILD_DIR}/${PERF_SCALING_EXEC_REL}" \
        "${OPTIMISATION_METHOD}" \
        "${REPEATS}" \
        "${RAW_CSV}" \
        "${PLOT_DIR}" \
        "${WARMUP_OPTION}" \
        "${SIZES[@]}"

    RAW_CSVS+=("${RAW_CSV}")
done

# Generate the combined comparison figures and summary tables from all raw CSV files.
MPLCONFIGDIR=/tmp/mpl_perf_graph \
python3 "${ROOT_DIR}/plot/plot_comparison_metrics.py" \
    "${COMPARISON_FIG_DIR}/combined" \
    "${RAW_CSVS[@]}"

# Print the output locations for the combined report-ready plots.
echo "==> Raw CSV directory: ${RAW_COMPARISON_DIR}"
echo "==> Combined plot directory: ${COMPARISON_FIG_DIR}/combined"
