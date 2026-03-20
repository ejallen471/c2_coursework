add_executable(
    simple_library_usage
    simple_library_usage.cpp
)

target_link_libraries(simple_library_usage PRIVATE cholesky_decomposition)
