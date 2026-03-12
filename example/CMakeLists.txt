add_executable(example_cholesky
    example_cholesky.cpp
)

target_link_libraries(example_cholesky
    PRIVATE
        mphil_dis_cholesky
)