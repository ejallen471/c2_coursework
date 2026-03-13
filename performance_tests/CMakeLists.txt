function(add_performance_tool target_name)
    add_executable(${target_name} ${ARGN})
    target_link_libraries(${target_name} PRIVATE mphil_dis_cholesky)

    # Older libstdc++ toolchains still require an explicit filesystem link library.
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9)
        target_link_libraries(${target_name} PRIVATE stdc++fs)
    endif()

    target_include_directories(${target_name}
        PRIVATE
            ${PROJECT_SOURCE_DIR}/src
            ${PROJECT_SOURCE_DIR}/performance_tests
    )
    target_compile_definitions(${target_name}
        PRIVATE
            MPHIL_PROJECT_SOURCE_DIR="${PROJECT_SOURCE_DIR}"
    )
endfunction()

find_package(LAPACK QUIET)

if(EXISTS "${PROJECT_SOURCE_DIR}/performance_tests/performance_driver.cpp")
    add_performance_tool(
        perf_driver
        ${PROJECT_SOURCE_DIR}/performance_tests/performance_driver.cpp
    )
endif()

add_performance_tool(
    perf_time
    perf_time.cpp
    perf_helpers.cpp
)

if(LAPACK_FOUND)
    target_link_libraries(perf_time PRIVATE ${LAPACK_LIBRARIES})
    target_compile_definitions(perf_time PRIVATE MPHIL_HAVE_LAPACK=1)
else()
    target_compile_definitions(perf_time PRIVATE MPHIL_HAVE_LAPACK=0)
endif()

add_performance_tool(
    perf_fixed_size_comparison
    perf_fixed_size_comparison.cpp
    perf_helpers.cpp
)

if(LAPACK_FOUND)
    target_link_libraries(perf_fixed_size_comparison PRIVATE ${LAPACK_LIBRARIES})
    target_compile_definitions(perf_fixed_size_comparison PRIVATE MPHIL_HAVE_LAPACK=1)
else()
    target_compile_definitions(perf_fixed_size_comparison PRIVATE MPHIL_HAVE_LAPACK=0)
endif()

add_performance_tool(
    perf_block_size_sweep
    perf_block_size_sweep.cpp
    perf_helpers.cpp
)

if(LAPACK_FOUND)
    target_link_libraries(perf_block_size_sweep PRIVATE ${LAPACK_LIBRARIES})
    target_compile_definitions(perf_block_size_sweep PRIVATE MPHIL_HAVE_LAPACK=1)
else()
    target_compile_definitions(perf_block_size_sweep PRIVATE MPHIL_HAVE_LAPACK=0)
endif()

add_performance_tool(
    perf_scaling
    perf_scaling.cpp
    perf_helpers.cpp
)

if(LAPACK_FOUND)
    target_link_libraries(perf_scaling PRIVATE ${LAPACK_LIBRARIES})
    target_compile_definitions(perf_scaling PRIVATE MPHIL_HAVE_LAPACK=1)
else()
    target_compile_definitions(perf_scaling PRIVATE MPHIL_HAVE_LAPACK=0)
endif()
