/*
perf_time.cpp

This file does the following

1. Read in an optimsation name and matrix size from the cmd line
2. Generate one SPD matrix of that size (using matrix.h and matrix.cpp)
3. Run the chosen cholesky implementation with timing (all through choleksy.cpp)
4. Print the time taken

*/

/**
 * @file perf_time.cpp
 * @brief Single-run benchmark driver for timing one Cholesky implementation on one generated matrix.
 */

#include "matrix.h"
#include "runtime_cholesky.h"

#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#if defined(MPHIL_HAVE_LAPACK) && MPHIL_HAVE_LAPACK
extern "C"
{
void dpotrf_(const char* uplo, const int* n, double* a, const int* lda, int* info);
}
#endif

namespace
{
/**
 * @brief Recover `log(det(A))` from the in-place Cholesky factor stored in a dense matrix.
 *
 * After factorisation, the returned storage contains the lower-triangular factor `L` such that
 * `A = L L^T`. The determinant is therefore the square of the product of the diagonal entries of
 * `L`, so `log(det(A)) = 2 * sum(log(L_ii))`.
 *
 * @param c Dense row-major matrix storage containing the Cholesky factor.
 * @param n Matrix dimension.
 * @return The recovered log-determinant.
 */
double logdet_from_factorised_storage(const std::vector<double>& c, int n)
{
    double sum = 0.0;

    for (int i = 0; i < n; ++i)
    {
        const std::size_t index =
            static_cast<std::size_t>(i) * static_cast<std::size_t>(n) + static_cast<std::size_t>(i);
        sum += std::log(c[index]);
    }

    return 2.0 * sum;
}

/**
 * @brief Compute the relative percentage difference between two scalar values.
 *
 * @param value Computed value under test.
 * @param reference Reference value used for comparison.
 * @return Relative difference expressed as a percentage.
 */
double relative_difference_percent(double value, double reference)
{
    const double scale = std::fabs(reference);
    if (scale == 0.0)
    {
        return (std::fabs(value) == 0.0) ? 0.0 : 100.0;
    }

    return 100.0 * std::fabs(value - reference) / scale;
}

#if defined(MPHIL_HAVE_LAPACK) && MPHIL_HAVE_LAPACK
/**
 * @brief Factorise a copy of the matrix with LAPACK and recover its log-determinant.
 *
 * @param c Dense row-major matrix storage containing the original SPD matrix.
 * @param n Matrix dimension.
 * @param logdet Output variable for the recovered reference log-determinant.
 * @return `true` if LAPACK successfully factorised the matrix, otherwise `false`.
 */
bool lapack_reference_logdet(std::vector<double> c, int n, double& logdet)
{
    const char uplo = 'L';
    const int lda = n;
    int info = 0;

    dpotrf_(&uplo, &n, c.data(), &lda, &info);
    if (info != 0)
    {
        return false;
    }

    logdet = logdet_from_factorised_storage(c, n);
    return true;
}
#endif
} // namespace

/**
 * @brief Generate one SPD matrix, run the requested Cholesky implementation, and print the elapsed time.
 *
 * Expected usage:
 * `perf_time <optimisation> <n>`
 *
 * @param argc Number of command-line arguments.
 * @param argv Command-line argument array containing the optimisation name and matrix size.
 * @return `0` on success, or a non-zero value if argument parsing or factorisation fails.
 */
int main(int argc, char* argv[])
{
    if (argc != 3)
    {
        std::cerr << "Usage: " << argv[0] << " <optimisation> <n>\n";
        return 1;
    }

    CholeskyVersion version;
    if (!parse_optimisation_name(argv[1], version))
    {
        std::cerr << "Error: unknown optimisation '" << argv[1] << "'\n";
        return 1;
    }

    const int n = std::atoi(argv[2]);
    if (n <= 0 || n > 100000)
    {
        std::cerr << "Error: n must be positive and at most 100000\n";
        return 1;
    }

    // Generate one dense SPD test matrix, keep one copy for the reference library calculation,
    // and factorise the working copy with the coursework implementation under test.
    const std::vector<double> original = make_generated_spd_matrix(n);
    std::vector<double> c = original;

    // run the requested implementation
    const double elapsed = run_cholesky_version(c.data(), n, version);

    // error checking case
    if (elapsed < 0.0)
    {
        std::cerr << "Error: factorisation failed with code " << elapsed << '\n';
        return 1;
    }

    // Recover the log-determinant from the returned factor so correctness can be compared against
    // a trusted LAPACK reference factorisation.
    const double computed_logdet = logdet_from_factorised_storage(c, n);

#if defined(MPHIL_HAVE_LAPACK) && MPHIL_HAVE_LAPACK
    double library_logdet = 0.0;
    if (!lapack_reference_logdet(original, n, library_logdet))
    {
        std::cerr << "Error: LAPACK reference factorisation failed\n";
        return 1;
    }

    const double relative_diff_pct =
        relative_difference_percent(computed_logdet, library_logdet);
#endif

    // Print the time, the library reference log-determinant, the log-determinant recovered from
    // the returned factor, and their relative percentage difference.
    std::cout << std::setprecision(16);
    std::cout << "optimisation=" << optimisation_name(version) << " n=" << n
              << " elapsed_seconds=" << elapsed;

#if defined(MPHIL_HAVE_LAPACK) && MPHIL_HAVE_LAPACK
    std::cout << " logdet_library=" << library_logdet
              << " logdet_factor=" << computed_logdet
              << " relative_difference_percent=" << relative_diff_pct;
#else
    std::cout << " logdet_factor=" << computed_logdet
              << " logdet_library=unavailable"
              << " relative_difference_percent=unavailable";
#endif

    std::cout << '\n';

    return 0;
}
