#!/usr/bin/env bash

# Exit on the first error, on unset variables, and on failures inside pipelines.
set -euo pipefail

# Figure out what directory this script is in
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Load shared project paths and executable locations from common_run_config.sh
source "${SCRIPT_DIR}/common_run_config.sh"

# Read the requested optimisation method from the first argument, defaulting to baseline.
OPTIMISATION_METHOD="${1:-baseline}"
# Read the number of repeats from the second argument, defaulting to 15.
REPEATS="${2:-15}"

# Allow the plotting conda environment name to be overridden by the caller.
CONDA_ENV_NAME="${CONDA_ENV_NAME:-coursework-plot}"

# Store the project-level conda environment definition file.
ENV_FILE="${ROOT_DIR}/environment.yml"

# Drop the first two arguments so any remaining arguments can be treated as matrix sizes.
shift $(( $# >= 2 ? 2 : $# ))

# If the caller did not provide sizes, use a small default sweep.
if [ "$#" -eq 0 ]; then
    SIZES=(64 128 256 512)
# Otherwise use all remaining command-line arguments as the size list.
else
    SIZES=("$@")
fi

# Use the shared prebuilt project directory.
BUILD_DIR="${SHARED_BUILD_DIR}"
WARMUP_OPTION="$(perf_warmup_cli_arg)"

# Write raw timing data to a CSV named after the optimisation method.
RAW_CSV="${RESULTS_RAW_DIR}/${OPTIMISATION_METHOD}_matrix_size_graph.csv"

# Write generated plots to a figure directory named after the optimisation method.
PLOT_DIR="${RESULTS_FIG_DIR}/${OPTIMISATION_METHOD}_matrix_size_graph"

# Ensure the output directories exist.
mkdir -p "${RESULTS_RAW_DIR}" "${RESULTS_FIG_DIR}"

# Print the chosen run settings before building or executing anything.
echo "==> Matrix size graph"

# Show which optimisation will be benchmarked.
echo "==> Optimisation: ${OPTIMISATION_METHOD}"

# Show how many repeats will be used per matrix size.
echo "==> Repeats: ${REPEATS}"

# Show the matrix sizes that will be swept.
echo "==> Sizes: ${SIZES[*]}"

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

# Load conda into this non-interactive shell so plotting can run with matplotlib available.
source "$(conda info --base)/etc/profile.d/conda.sh"

# If the plotting environment does not exist yet, create it from environment.yml.
if ! conda env list | awk 'NF && $1 !~ /^#/ { print $1 }' | grep -Fxq "${CONDA_ENV_NAME}"; then
    echo "==> Creating conda environment: ${CONDA_ENV_NAME}"
    conda env create --yes -n "${CONDA_ENV_NAME}" -f "${ENV_FILE}"
fi

# Activate the conda environment used for plotting before launching the scaling driver.
conda activate "${CONDA_ENV_NAME}"

# Give matplotlib a writable cache directory, then run the scaling driver.
# Argument 1: optimisation method to benchmark.
# Argument 2: repeat count per matrix size.
# Argument 3: output CSV path for raw timing data.
# Argument 4: output directory for plots and summaries.
# Remaining arguments: matrix sizes to sweep over.
MPLCONFIGDIR=/tmp/mpl_perf_graph \
"${BUILD_DIR}/${PERF_SCALING_EXEC_REL}" \
    "${OPTIMISATION_METHOD}" \
    "${REPEATS}" \
    "${RAW_CSV}" \
    "${PLOT_DIR}" \
    "${WARMUP_OPTION}" \
    "${SIZES[@]}"

# Print the location of the raw CSV output.
echo "==> Raw CSV: ${RAW_CSV}"

# Print the location of the generated plots and summaries.
echo "==> Plot directory: ${PLOT_DIR}"
