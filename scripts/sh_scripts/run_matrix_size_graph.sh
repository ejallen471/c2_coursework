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
ensure_shared_release_executable "${RUN_CHOLESKY_EXEC_REL}"
PLOT_PYTHON="$(resolve_plot_python)"

# Run the scaling benchmark first so the raw CSV exists before plotting.
"${BUILD_DIR}/${RUN_CHOLESKY_EXEC_REL}" \
    scaling \
    "${OPTIMISATION_METHOD}" \
    "${REPEATS}" \
    "${RAW_CSV}" \
    "${SIZES[@]}"

# Give matplotlib a writable cache directory, then plot from the generated CSV.
MPLCONFIGDIR=/tmp/mpl_perf_graph \
"${PLOT_PYTHON}" "${ROOT_DIR}/plot/plot_metrics.py" \
    "${RAW_CSV}" \
    "${PLOT_DIR}"

# Print the location of the raw CSV output.
echo "==> Raw CSV: ${RAW_CSV}"

# Print the location of the generated plots and summaries.
echo "==> Plot directory: ${PLOT_DIR}"
