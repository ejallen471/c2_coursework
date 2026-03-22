/**
 * @file example.cpp
 * @brief Small example showing how to call the public Cholesky library API.
 */

#include "cholesky_decomposition.h"

#include <iomanip>
#include <iostream>
#include <vector>

namespace
{
/**
 * @brief Print a square row-major matrix.
 *
 * @param matrix Matrix values in row-major order.
 * @param n Matrix dimension.
 */
void print_matrix(const std::vector<double>& matrix, int n)
{
    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            std::cout << std::setw(10) << matrix[static_cast<std::size_t>(i * n + j)] << ' ';
        }
        std::cout << '\n';
    }
}
} // namespace

int main()
{
    // A small symmetric positive definite matrix stored in row-major order.
    std::vector<double> matrix = {
        4.0,
        12.0,
        -16.0,
        12.0,
        37.0,
        -43.0,
        -16.0,
        -43.0,
        98.0,
    };

    const int n = 3;

    std::cout << "Input matrix A:\n";
    print_matrix(matrix, n);

    CholeskyRuntimeOptions options;
    options.block_size = 2;
    options.thread_count = 1;

    const double elapsed = timed_cholesky_factorisation_versioned_configured(
        matrix.data(), n, CholeskyVersion::BlockedTileKernels, options);

    if (elapsed < 0.0)
    {
        std::cerr << "Factorisation failed with timing/status code " << elapsed << '\n';
        return 1;
    }

    std::cout << "\nFactorised matrix storage:\n";
    print_matrix(matrix, n);

    std::cout << "\nElapsed time (s): " << elapsed << '\n';
    std::cout << "The upper triangle contains the Cholesky factor R such that A = R^T R.\n";

    return 0;
}
