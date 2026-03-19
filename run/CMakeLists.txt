add_executable(
    run_cholesky
    run_cholesky.cpp
)

target_link_libraries(run_cholesky PRIVATE coursework_perf_support)

target_include_directories(run_cholesky
    PRIVATE
        ${PROJECT_SOURCE_DIR}/src
        ${PROJECT_SOURCE_DIR}/performance_tests
)
