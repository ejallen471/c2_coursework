add_library(mphil_dis_cholesky
    mphil_dis_cholesky.cpp
    cholesky_baseline.cpp
    cholesky_lower_triangle_only.cpp
    cholesky_inline_mirror.cpp
    cholesky_loop_cleanup.cpp
    cholesky_access_pattern_aware.cpp
    cholesky_cache_blocked.cpp
    cholesky_vectorisation.cpp
    cholesky_blocked_vectorised.cpp
    cholesky_openmp_1.cpp
    cholesky_openmp_2.cpp
    cholesky_openmp_3.cpp
    ${PROJECT_SOURCE_DIR}/utils/timer.cpp
    ${PROJECT_SOURCE_DIR}/utils/matrix.cpp
)

target_include_directories(mphil_dis_cholesky
    PUBLIC
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/utils
)

find_package(OpenMP QUIET)
if(OpenMP_CXX_FOUND)
    target_link_libraries(mphil_dis_cholesky PUBLIC OpenMP::OpenMP_CXX)
    target_compile_definitions(mphil_dis_cholesky PUBLIC MPHIL_HAVE_OPENMP=1)
else()
    target_compile_definitions(mphil_dis_cholesky PUBLIC MPHIL_HAVE_OPENMP=0)
endif()
