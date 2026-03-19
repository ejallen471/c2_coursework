find_package(LAPACK REQUIRED)

add_library(
    coursework_perf_support
    01_perf_method_compare.cpp
    02_perf_thread_count_sweep.cpp
    03_perf_block_size_sweep.cpp
    04_perf_matrix_size_sweep.cpp
    perf_helpers.cpp
    runtime_cholesky.cpp
)

target_link_libraries(coursework_perf_support PUBLIC cholesky_decomposition)

target_include_directories(coursework_perf_support
    PUBLIC
        ${PROJECT_SOURCE_DIR}/src
        ${PROJECT_SOURCE_DIR}/performance_tests
)

# Older libstdc++ toolchains still require an explicit filesystem link library.
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9)
    target_link_libraries(coursework_perf_support PUBLIC stdc++fs)
endif()

target_link_libraries(coursework_perf_support PUBLIC ${LAPACK_LIBRARIES})
