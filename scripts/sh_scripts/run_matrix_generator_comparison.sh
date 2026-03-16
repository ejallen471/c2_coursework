#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common_run_config.sh"

MATRIX_SIZE="${1:-1000}"
RAW_CSV="${2:-${RESULTS_RAW_DIR}/matrix_generator_comparison_n${MATRIX_SIZE}.csv}"

mkdir -p "$(dirname "${RAW_CSV}")"

echo "==> Matrix generator comparison"
echo "==> Matrix size: ${MATRIX_SIZE}"
echo "==> Raw CSV: ${RAW_CSV}"

bash "${SHARED_BUILD_SCRIPT}"

"${SHARED_BUILD_DIR}/run/compare_matrix_generators" \
    "${MATRIX_SIZE}" \
    "${RAW_CSV}"
