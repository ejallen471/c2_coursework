#!/bin/bash
#SBATCH --job-name=st_scaling_all
#SBATCH --account=MPHIL-DIS-SL2-CPU
#SBATCH --partition=sapphire
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=06:00:00
#SBATCH --output=logs/single_thread_scaling_all/slurm-%x-%j.out
#SBATCH --error=logs/single_thread_scaling_all/slurm-%x-%j.err

set -eo pipefail

ROOT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${ROOT_DIR}"

REPEATS="${1:-5}"

shift $(( $# >= 1 ? 1 : $# ))

if [ "$#" -eq 0 ]; then
    SIZES=(512 1024 2000 4096 6000 8192 10000 15000)
else
    SIZES=("$@")
fi

METHODS=(
    baseline
    lower_triangle
    upper_triangle
    contiguous_access
    cache_blocked_1
    cache_blocked_2
)

mkdir -p results/raw results/figures

module purge
module load cmake
module load gcc
module load python

source ~/venvs/c2_coursework/bin/activate

bash scripts/sh_scripts/build.sh

RAW_CSV="results/raw/all_methods_graph.csv"
PLOT_DIR="results/figures/all_methods_graph"

TEMP_RAW_DIR="$(mktemp -d "${TMPDIR:-/tmp}/all_methods_graph.XXXXXX")"
trap 'rm -rf "${TEMP_RAW_DIR}"' EXIT

FIRST_CSV=1

for METHOD in "${METHODS[@]}"; do
    METHOD_RAW_CSV="${TEMP_RAW_DIR}/${METHOD}.csv"
    echo "==> Running ${METHOD}"

    ./build/run/run_cholesky \
        scaling \
        "${METHOD}" \
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

MPLCONFIGDIR=/tmp/mpl_perf_graph \
python plot/plot_comparison_metrics.py \
    "${PLOT_DIR}" \
    "${RAW_CSV}"

echo "raw_csv=\"${RAW_CSV}\""
echo "plot_dir=\"${PLOT_DIR}\""
