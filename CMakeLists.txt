cmake_minimum_required(VERSION 3.16)

project(C2_COURSEWORK LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type (Debug, Release, RelWithDebInfo, MinSizeRel)" FORCE)
endif()

option(MPHIL_ENABLE_DEBUG_CHECKS "Enable additional debug-only runtime checks" ON)
option(MPHIL_ENABLE_DOXYGEN "Enable the Doxygen docs target when Doxygen is available" ON)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
    add_compile_options(
        "$<$<CONFIG:Debug>:-O0>"
        "$<$<CONFIG:Debug>:-g3>"
        "$<$<CONFIG:Debug>:-fno-omit-frame-pointer>"
        "$<$<CONFIG:Debug>:-Wall>"
        "$<$<CONFIG:Debug>:-Wextra>"
        "$<$<CONFIG:Debug>:-Wpedantic>"
    )
endif()

if(MPHIL_ENABLE_DEBUG_CHECKS)
    add_compile_definitions(
        "$<$<CONFIG:Debug>:MPHIL_DEBUG_CHECKS=1>"
        "$<$<CONFIG:Debug>:_GLIBCXX_ASSERTIONS>"
    )
endif()

include(CTest)

add_subdirectory(src)
add_subdirectory(performance_tests)
add_subdirectory(run)

if(BUILD_TESTING)
    add_subdirectory(test)
endif()

if(MPHIL_ENABLE_DOXYGEN)
    find_package(Doxygen QUIET)

    if(DOXYGEN_FOUND)
        set(DOXYGEN_PROJECT_NAME "${PROJECT_NAME}")
        set(DOXYGEN_PROJECT_BRIEF "Dense Cholesky factorisation coursework benchmarks and implementations.")
        set(DOXYGEN_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/docs")
        set(DOXYGEN_GENERATE_HTML YES)
        set(DOXYGEN_GENERATE_LATEX NO)
        set(DOXYGEN_JAVADOC_AUTOBRIEF YES)
        set(DOXYGEN_MARKDOWN_SUPPORT YES)
        set(DOXYGEN_RECURSIVE YES)
        set(DOXYGEN_EXTRACT_ALL NO)
        set(DOXYGEN_EXTRACT_PRIVATE YES)
        set(DOXYGEN_EXTRACT_STATIC YES)
        set(DOXYGEN_SOURCE_BROWSER YES)
        set(DOXYGEN_FULL_PATH_NAMES NO)
        set(DOXYGEN_STRIP_FROM_PATH "${PROJECT_SOURCE_DIR}")
        set(DOXYGEN_USE_MDFILE_AS_MAINPAGE "${PROJECT_SOURCE_DIR}/README.md")
        set(DOXYGEN_FILE_PATTERNS "*.h" "*.cpp" "*.md")
        set(DOXYGEN_EXCLUDE_PATTERNS "*/build/*" "*/build_openmp/*" "*/results/*")

        doxygen_add_docs(
            docs
            ${PROJECT_SOURCE_DIR}/README.md
            ${PROJECT_SOURCE_DIR}/docs
            ${PROJECT_SOURCE_DIR}/include
            ${PROJECT_SOURCE_DIR}/src
            ${PROJECT_SOURCE_DIR}/utils
            ${PROJECT_SOURCE_DIR}/run
            ${PROJECT_SOURCE_DIR}/performance_tests
            ${PROJECT_SOURCE_DIR}/test
            COMMENT "Generate Doxygen documentation"
        )
    else()
        message(STATUS "Doxygen not found. Install it to enable the 'docs' target.")
    endif()
endif()
