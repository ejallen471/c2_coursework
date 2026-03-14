function(add_coursework_test test_name)
    add_executable(${test_name} ${ARGN})
    target_link_libraries(${test_name} PRIVATE mphil_dis_cholesky)
    target_include_directories(${test_name}
        PRIVATE
            ${PROJECT_SOURCE_DIR}/src
            ${PROJECT_SOURCE_DIR}/performance_tests
    )
    add_test(NAME ${test_name} COMMAND ${test_name})
endfunction()

add_coursework_test(test_small_case test_small_case.cpp test_helpers.cpp)
add_coursework_test(test_invalid_input test_invalid_input.cpp)
add_coursework_test(test_identity test_identity.cpp test_helpers.cpp)
add_coursework_test(test_diagonal_matrix test_diagonal_matrix.cpp test_helpers.cpp)
add_coursework_test(test_reconstruction test_reconstruction.cpp test_helpers.cpp)
add_coursework_test(test_logdet test_logdet.cpp test_helpers.cpp)
add_coursework_test(test_generated_spd test_generated_spd.cpp test_helpers.cpp)
add_coursework_test(test_repeatability test_repeatability.cpp test_helpers.cpp)
add_coursework_test(test_matrix_generation test_matrix_generation.cpp test_helpers.cpp)
add_coursework_test(test_version_correctness test_version_correctness.cpp test_helpers.cpp)

add_executable(test_perf_time_output test_perf_time_output.cpp)
add_test(NAME test_perf_time_output COMMAND test_perf_time_output $<TARGET_FILE:run_cholesky>)
