find_package(LAPACK QUIET)

add_library(
    coursework_perf_support
    perf_time.cpp
    perf_fixed_size_comparison.cpp
    perf_block_size_sweep.cpp
    perf_scaling.cpp
    perf_helpers.cpp
)

target_link_libraries(coursework_perf_support PUBLIC mphil_dis_cholesky)

target_include_directories(coursework_perf_support
    PUBLIC
        ${PROJECT_SOURCE_DIR}/src
        ${PROJECT_SOURCE_DIR}/performance_tests
)

# Older libstdc++ toolchains still require an explicit filesystem link library.
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9)
    target_link_libraries(coursework_perf_support PUBLIC stdc++fs)
endif()

if(LAPACK_FOUND)
    target_link_libraries(coursework_perf_support PUBLIC ${LAPACK_LIBRARIES})
    target_compile_definitions(coursework_perf_support PUBLIC MPHIL_HAVE_LAPACK=1)
else()
    target_compile_definitions(coursework_perf_support PUBLIC MPHIL_HAVE_LAPACK=0)
endif()
