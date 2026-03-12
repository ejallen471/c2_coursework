add_library(mphil_dis_cholesky
    mphil_dis_cholesky.cpp
    cholesky_versions.cpp
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

if(MPHIL_CHOLESKY_VERSION STREQUAL "Baseline")
    target_compile_definitions(mphil_dis_cholesky PRIVATE MPHIL_CHOLESKY_VERSION_ID=0)
else()
    message(FATAL_ERROR "Unsupported MPHIL_CHOLESKY_VERSION='${MPHIL_CHOLESKY_VERSION}'")
endif()
