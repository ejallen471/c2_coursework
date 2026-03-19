add_library(cholesky_decomposition
    cholesky_decomposition.cpp
    cholesky_versions.cpp
    timer.cpp
)

target_include_directories(cholesky_decomposition
    PUBLIC
        ${PROJECT_SOURCE_DIR}/include
)
