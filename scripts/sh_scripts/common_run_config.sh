#!/usr/bin/env bash

# Resolve the directory that contains this shared config script.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Resolve the repository root by going two levels up from the script directory.
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Store the directory where raw CSV benchmark outputs should be written.
RESULTS_RAW_DIR="${ROOT_DIR}/results/raw"

# Store the directory where generated figures should be written.
RESULTS_FIG_DIR="${ROOT_DIR}/results/figures"

# Store the shared build directory used by the normal build and run workflows.
SHARED_BUILD_DIR="${ROOT_DIR}/build"

# Store the optimised CMake build type used by the normal shell workflows.
SHARED_BUILD_TYPE="${SHARED_BUILD_TYPE:-Release}"

# Store the path to the shared build script used to populate the optimised build directory.
SHARED_BUILD_SCRIPT="${SCRIPT_DIR}/build.sh"

# Store the OpenMP-specific build directory used by the OpenMP shell workflows.
OPENMP_BUILD_DIR="${ROOT_DIR}/build_openmp"

# Store the path to the OpenMP build script used to populate the OpenMP build directory.
OPENMP_BUILD_SCRIPT="${SCRIPT_DIR}/build_openmp.sh"

# Relative path to the unified benchmark runner inside a build directory.
RUN_CHOLESKY_EXEC_REL="run/run_cholesky"

# Allow callers to disable the optional local sanity benchmark by setting an environment variable.
RUN_LOCAL_SANITY_PERF="${RUN_LOCAL_SANITY_PERF:-1}"

# Default matrix size for local sanity benchmark runs when not overridden.
LOCAL_SANITY_N="${LOCAL_SANITY_N:-128}"

# Default repeat count for local sanity benchmark runs when not overridden.
LOCAL_SANITY_REPEATS="${LOCAL_SANITY_REPEATS:-3}"

binary_matches_current_platform() {
    local exec_path="$1"
    local file_output=""

    if [ ! -x "${exec_path}" ]; then
        return 1
    fi

    if ! command -v file >/dev/null 2>&1; then
        return 0
    fi

    file_output="$(file -b "${exec_path}" 2>/dev/null || true)"

    case "$(uname -s)" in
        Darwin)
            [[ "${file_output}" == *"Mach-O"* ]]
            ;;
        Linux)
            [[ "${file_output}" == *"ELF"* ]]
            ;;
        *)
            return 0
            ;;
    esac
}

ensure_release_executable_in_build() {
    local build_dir="$1"
    local exec_rel="$2"
    local build_script="$3"
    local exec_path="${build_dir}/${exec_rel}"
    local cache_path="${build_dir}/CMakeCache.txt"
    local configured_build_type=""
    local needs_rebuild=0

    if [ -f "${cache_path}" ]; then
        configured_build_type="$(awk -F= '/^CMAKE_BUILD_TYPE:STRING=/{print $2; exit}' "${cache_path}")"
    fi

    if [ ! -x "${exec_path}" ] || [ "${configured_build_type}" != "${SHARED_BUILD_TYPE}" ]; then
        needs_rebuild=1
    fi

    if [ "${needs_rebuild}" -eq 0 ] && ! binary_matches_current_platform "${exec_path}"; then
        echo "==> Existing executable in ${build_dir} is incompatible with $(uname -s); rebuilding"
        rm -rf "${build_dir}"
        needs_rebuild=1
    fi

    if [ "${needs_rebuild}" -eq 1 ]; then
        echo "==> Ensuring ${SHARED_BUILD_TYPE} build in ${build_dir}"
        bash "${build_script}"
    fi

    if [ ! -x "${exec_path}" ]; then
        echo "Error: missing executable ${exec_path}" >&2
        return 1
    fi
}

ensure_shared_release_executable() {
    local exec_rel="$1"
    ensure_release_executable_in_build "${SHARED_BUILD_DIR}" "${exec_rel}" "${SHARED_BUILD_SCRIPT}"
}

resolve_plot_python() {
    if [ -n "${PLOT_PYTHON:-}" ]; then
        if [ ! -x "${PLOT_PYTHON}" ]; then
            echo "Error: PLOT_PYTHON is set but is not executable: ${PLOT_PYTHON}" >&2
            return 1
        fi

        printf '%s\n' "${PLOT_PYTHON}"
        return 0
    fi

    if command -v python >/dev/null 2>&1; then
        command -v python
        return 0
    fi

    if command -v python3 >/dev/null 2>&1; then
        command -v python3
        return 0
    fi

    echo "Error: python was not found on PATH. Activate your virtual environment and install requirements-plot.txt." >&2
    return 1
}
