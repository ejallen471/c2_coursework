#!/bin/bash
#SBATCH -J chol-allgraph
#SBATCH -A MPHIL-DIS-SL2-CPU
#SBATCH -o logs/chol_allgraph_%j.out
#SBATCH -e logs/chol_allgraph_%j.err
#SBATCH -p icelake
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --time=05:00:00

set -euo pipefail

# Run this script from the repository root with:
# sbatch scripts/slurm_scripts/run_all_methods_graph.slurm <repeats> <n1> [n2 ...]

# Resolve the repository root from the directory where sbatch was launched.
ROOT_DIR="${SLURM_SUBMIT_DIR}"

# Read the repeat count from the first argument, defaulting to 3 so the full
# all-method sweep stays within the 5-hour wall-clock cap on CSD3.
REPEATS="${1:-3}"
PERF_WARMUP="${PERF_WARMUP:-1}"

# Allow the plotting conda environment name to be overridden by the caller.
CONDA_ENV_NAME="${CONDA_ENV_NAME:-coursework-plot}"

# Store the project-level conda environment definition file.
ENV_FILE="${ROOT_DIR}/environment.yml"

# Drop the first argument so the remaining arguments can be treated as matrix sizes.
shift $(( $# >= 1 ? 1 : $# ))

# If no matrix sizes were provided, use a default sweep.
if [ "$#" -eq 0 ]; then
    SIZES=(2000 4000 6000 8000 10000)
else
    SIZES=("$@")
fi

# Use the shared prebuilt build directory.
BUILD_DIR="${ROOT_DIR}/build"

# Store raw CSV files and figures in dedicated comparison directories.
RESULTS_RAW_DIR="${ROOT_DIR}/results/raw"
RESULTS_FIG_DIR="${ROOT_DIR}/results/figures"
RAW_COMPARISON_DIR="${RESULTS_RAW_DIR}/all_methods_graph"
COMPARISON_FIG_DIR="${RESULTS_FIG_DIR}/all_methods_graph"
LOG_DIR="${ROOT_DIR}/logs"

# Relative path to the scaling executable inside the build directory.
PERF_SCALING_EXEC_REL="performance_tests/perf_scaling"

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

# Ensure the output directories exist.
mkdir -p "${RAW_COMPARISON_DIR}" "${COMPARISON_FIG_DIR}" "${LOG_DIR}"

# Load the standard CSD3 runtime and plotting environment.
module purge
module load gcc
module load miniconda/3

# Stop early with a clear error if the prebuilt executable is missing.
if [ ! -x "${BUILD_DIR}/${PERF_SCALING_EXEC_REL}" ]; then
    echo "Error: missing executable ${BUILD_DIR}/${PERF_SCALING_EXEC_REL}" >&2
    echo "Build the project first with scripts/sh_scripts/build.sh" >&2
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

# Activate the plotting environment.
conda activate "${CONDA_ENV_NAME}"

# Print the requested run configuration.
echo "==> Slurm all-method matrix size comparison"
echo "==> Repeats: ${REPEATS}"
echo "==> Sizes: ${SIZES[*]}"
echo "==> Methods: ${OPTIMISATION_METHODS[*]}"

case "${PERF_WARMUP}" in
    1|true|TRUE|on|ON|yes|YES)
        WARMUP_OPTION="--warmup"
        ;;
    0|false|FALSE|off|OFF|no|NO)
        WARMUP_OPTION="--no-warmup"
        ;;
    *)
        echo "Error: PERF_WARMUP must be one of 0/1, true/false, on/off, yes/no" >&2
        exit 1
        ;;
esac

# Collect the generated raw CSV files so the combined plotter can consume them.
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
