add_library(mphil_dis_cholesky
    mphil_dis_cholesky.cpp
    00_cholesky_baseline.cpp
    01_cholesky_lower_triangle.cpp
    02_cholesky_upper_triangle.cpp
    03_cholesky_contiguous_access.cpp
    04_cholesky_cache_blocked_1.cpp
    05_cholesky_cache_blocked_2.cpp
    06_cholesky_openmp_1.cpp
    07_cholesky_openmp_2.cpp
    08_cholesky_openmp_3.cpp
    09_cholesky_openmp_4.cpp
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
