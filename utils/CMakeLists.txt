add_library(mphil_dis_cholesky
    mphil_dis_cholesky.cpp
    cholesky_versions.cpp
    timer.cpp
)

target_include_directories(mphil_dis_cholesky
    PUBLIC
        ${PROJECT_SOURCE_DIR}/include
)
