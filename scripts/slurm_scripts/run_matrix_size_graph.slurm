#!/bin/bash
#SBATCH -J chol-graph
#SBATCH -A MPHIL-DIS-SL2-CPU
#SBATCH -o logs/chol_graph_%j.out
#SBATCH -e logs/chol_graph_%j.err
#SBATCH -p icelake
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --time=05:00:00

set -euo pipefail

# Run this script from the repository root with:
# sbatch scripts/slurm_scripts/run_matrix_size_graph.slurm <optimisation> <repeats> <n1> [n2 ...]

# Resolve the repository root from the directory where sbatch was launched.
ROOT_DIR="${SLURM_SUBMIT_DIR}"

# Read the optimisation method from the first argument, defaulting to baseline.
OPTIMISATION_METHOD="${1:-baseline}"

# Read the repeats per matrix size from the second argument, defaulting to 15.
REPEATS="${2:-15}"
PERF_WARMUP="${PERF_WARMUP:-1}"

# Allow the plotting conda environment name to be overridden by the caller.
CONDA_ENV_NAME="${CONDA_ENV_NAME:-coursework-plot}"

# Store the project-level conda environment definition file.
ENV_FILE="${ROOT_DIR}/environment.yml"

# Drop the first two arguments so the remaining arguments can be treated as matrix sizes.
shift $(( $# >= 2 ? 2 : $# ))

# If no matrix sizes were provided, use a small default sweep.
if [ "$#" -eq 0 ]; then
    SIZES=(64 128 256 512)
else
    SIZES=("$@")
fi

# Use the shared build directory created ahead of time by the build script.
BUILD_DIR="${ROOT_DIR}/build"

# Store raw CSV results under the repository results directory.
RESULTS_RAW_DIR="${ROOT_DIR}/results/raw"

# Store generated figures under the repository results directory.
RESULTS_FIG_DIR="${ROOT_DIR}/results/figures"

# Store logs under the repository root.
LOG_DIR="${ROOT_DIR}/logs"

# Write raw timing data to a CSV named after the optimisation method.
RAW_CSV="${RESULTS_RAW_DIR}/${OPTIMISATION_METHOD}_matrix_size_graph.csv"

# Write plots and processed summaries to a directory named after the optimisation method.
PLOT_DIR="${RESULTS_FIG_DIR}/${OPTIMISATION_METHOD}_matrix_size_graph"

# Relative path to the scaling executable inside the build directory.
PERF_SCALING_EXEC_REL="performance_tests/perf_scaling"

# Ensure the results, figure, and log directories exist.
mkdir -p "${RESULTS_RAW_DIR}" "${RESULTS_FIG_DIR}" "${LOG_DIR}"

# Load the standard CSD3 toolchain modules needed to run the prebuilt executable and plot.
module purge
module load gcc
module load miniconda/3

# Print the requested run configuration.
echo "==> Slurm matrix size graph"
echo "==> Root directory: ${ROOT_DIR}"
echo "==> Build directory: ${BUILD_DIR}"
echo "==> Optimisation: ${OPTIMISATION_METHOD}"
echo "==> Repeats: ${REPEATS}"
echo "==> Sizes: ${SIZES[*]}"
echo "==> Conda environment: ${CONDA_ENV_NAME}"

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

# Stop early with a clear error if the prebuilt executable is missing.
if [ ! -x "${BUILD_DIR}/${PERF_SCALING_EXEC_REL}" ]; then
    echo "Error: missing executable ${BUILD_DIR}/${PERF_SCALING_EXEC_REL}" >&2
    echo "Build the project first with scripts/sh_scripts/build.sh" >&2
    exit 1
fi

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

# If the plotting environment does not exist yet, create it from environment.yml.
if ! conda env list | awk 'NF && $1 !~ /^#/ { print $1 }' | grep -Fxq "${CONDA_ENV_NAME}"; then
    echo "==> Creating conda environment: ${CONDA_ENV_NAME}"
    conda env create --yes -n "${CONDA_ENV_NAME}" -f "${ENV_FILE}"
fi

# Activate the conda environment used for plotting.
conda activate "${CONDA_ENV_NAME}"

# Run the scaling executable and let it write the raw CSV, summary CSV, and plots.
MPLCONFIGDIR=/tmp/mpl_perf_graph \
"${BUILD_DIR}/${PERF_SCALING_EXEC_REL}" \
    "${OPTIMISATION_METHOD}" \
    "${REPEATS}" \
    "${RAW_CSV}" \
    "${PLOT_DIR}" \
    "${WARMUP_OPTION}" \
    "${SIZES[@]}"

# Print the locations of the generated outputs.
echo "==> Raw CSV: ${RAW_CSV}"
echo "==> Plot directory: ${PLOT_DIR}"
