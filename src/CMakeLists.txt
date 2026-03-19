add_library(cholesky_decomposition
    cholesky_decomposition.cpp
    00_cholesky_baseline.cpp
    01_cholesky_lower_triangle.cpp
    02_cholesky_upper_triangle.cpp
    03_cholesky_contiguous_access.cpp
    04_cholesky_blocked_tile_kernels.cpp
    05_cholesky_blocked_tile_kernels_unrolled.cpp
    06_cholesky_openmp_row_parallel_unblocked.cpp
    07_cholesky_openmp_tile_parallel_blocked.cpp
    08_cholesky_openmp_block_row_parallel.cpp
    09_cholesky_openmp_tile_list_parallel.cpp
    10_cholesky_openmp_task_dag_blocked.cpp
    ${PROJECT_SOURCE_DIR}/utils/timer.cpp
    ${PROJECT_SOURCE_DIR}/utils/matrix.cpp
)

target_include_directories(cholesky_decomposition
    PUBLIC
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/utils
)

# Probe for OpenMP so the library can link the compiler's threading runtime when the
# OpenMP kernels are enabled, while still allowing a non-OpenMP build to configure cleanly.
find_package(OpenMP QUIET)
if(OpenMP_CXX_FOUND)
    # Link the imported OpenMP target publicly so any consumer of `cholesky_decomposition`
    # inherits the same OpenMP runtime that the kernel objects were compiled against.
    target_link_libraries(cholesky_decomposition PUBLIC OpenMP::OpenMP_CXX)
endif()
