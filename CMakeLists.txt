cmake_minimum_required(VERSION 3.16)

project(C2_COURSEWORK LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(MPHIL_CHOLESKY_VERSION "Baseline" CACHE STRING "Cholesky implementation to build")
set_property(CACHE MPHIL_CHOLESKY_VERSION PROPERTY STRINGS
    Baseline
    LowerTriangleOnly
    InlineMirror
    LoopCleanup
    AccessPatternAware
    CacheBlocked
    VectorFriendly
    BlockedVectorised
    OpenMP1
    OpenMP2
    OpenMP3
)

if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type (Debug, Release, RelWithDebInfo, MinSizeRel)" FORCE)
endif()

option(MPHIL_ENABLE_DEBUG_CHECKS "Enable additional debug-only runtime checks" ON)

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
add_subdirectory(example)
add_subdirectory(performance_tests)

if(BUILD_TESTING)
    add_subdirectory(test)
endif()
