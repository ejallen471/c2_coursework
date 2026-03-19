find_package(Python3 COMPONENTS Interpreter REQUIRED)
find_package(GTest REQUIRED)
include(GoogleTest)

add_library(coursework_test_support
    test_suite_helpers.cpp
)

target_link_libraries(coursework_test_support PUBLIC coursework_perf_support)
target_include_directories(coursework_test_support
    PUBLIC
        ${PROJECT_SOURCE_DIR}/src
        ${PROJECT_SOURCE_DIR}/performance_tests
        ${CMAKE_CURRENT_SOURCE_DIR}
)
target_compile_definitions(coursework_test_support
    PUBLIC
        COURSEWORK_SOURCE_ROOT="${PROJECT_SOURCE_DIR}"
        COURSEWORK_BINARY_ROOT="${PROJECT_BINARY_DIR}"
        COURSEWORK_PYTHON_EXECUTABLE="${Python3_EXECUTABLE}"
)

set(COURSEWORK_GTEST_BINARY_DIRECTORY
    "${CMAKE_CURRENT_BINARY_DIR}/gtest_discovery")
set(COURSEWORK_GTEST_WORKING_DIRECTORY
    "${COURSEWORK_GTEST_BINARY_DIRECTORY}/runtime")
file(MAKE_DIRECTORY "${COURSEWORK_GTEST_WORKING_DIRECTORY}")

function(add_coursework_gtest target_name labels)
    add_executable(${target_name} ${ARGN})
    target_link_libraries(
        ${target_name}
        PRIVATE
            coursework_test_support
            GTest::gtest_main
    )
    target_include_directories(${target_name}
        PRIVATE
            ${PROJECT_SOURCE_DIR}/src
            ${PROJECT_SOURCE_DIR}/performance_tests
            ${PROJECT_SOURCE_DIR}/test
    )
    set_target_properties(${target_name} PROPERTIES
        BUILD_RPATH "$<TARGET_FILE_DIR:GTest::gtest_main>;$<TARGET_FILE_DIR:GTest::gtest>"
        RUNTIME_OUTPUT_DIRECTORY "${COURSEWORK_GTEST_BINARY_DIRECTORY}"
    )

    gtest_discover_tests(
        ${target_name}
        TEST_PREFIX "${target_name}."
        WORKING_DIRECTORY ${COURSEWORK_GTEST_WORKING_DIRECTORY}
        DISCOVERY_MODE POST_BUILD
        PROPERTIES LABELS "${labels}"
    )
endfunction()

add_coursework_gtest(
    coursework_unit_gtests
    "unit"
    ${PROJECT_SOURCE_DIR}/test/test_unit_gtest.cpp
)

add_coursework_gtest(
    coursework_correctness_gtests
    "correctness"
    ${PROJECT_SOURCE_DIR}/test/test_correctness_gtest.cpp
)

add_coursework_gtest(
    coursework_openmp_gtests
    "openmp"
    ${PROJECT_SOURCE_DIR}/test/test_openmp_gtest.cpp
)

add_coursework_gtest(
    coursework_regression_gtests
    "regression"
    ${PROJECT_SOURCE_DIR}/test/test_regression_gtest.cpp
)

add_coursework_gtest(
    coursework_integration_cpp_gtests
    "integration"
    ${PROJECT_SOURCE_DIR}/test/test_integration_gtest.cpp
)

add_coursework_gtest(
    coursework_cli_gtests
    "integration"
    ${PROJECT_SOURCE_DIR}/test/test_cli_gtest.cpp
)

add_coursework_gtest(
    coursework_csv_contract_gtests
    "integration"
    ${PROJECT_SOURCE_DIR}/test/test_csv_contract_gtest.cpp
)

add_coursework_gtest(
    coursework_plot_gtests
    "plot"
    ${PROJECT_SOURCE_DIR}/test/test_plot_gtest.cpp
)
