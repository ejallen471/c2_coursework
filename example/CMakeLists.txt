add_executable(
    simple_library_usage
    example.cpp
)

target_link_libraries(simple_library_usage PRIVATE cholesky_decomposition)
