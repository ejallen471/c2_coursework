/**
 * @file matrix.cpp
 * @brief Implementations for SPD matrix construction and validation helpers.
 */

/*
This code builds dense symmetric positive definite (SPD) matrices.

We generate a random symmetric matrix and then enforce strict diagonal dominance:

    a_ii > sum_{j != i} |a_ij|

For symmetric matrices, this is a sufficient condition for SPD.
*/

#include "matrix.h"

#include <cassert>
#include <cmath>
#include <cstddef>
#include <random>

namespace
{
    constexpr double kDefaultAdder = 1.0e-3;   // Compile time constant to add to the diagonal for
                                               // numerical stability and ensure SPD condition
    constexpr double kPositiveFloor = 1.0e-12; // Compile time constant, Used to decide whether a
                                               // parameter should be treated as positive or invalid.
    constexpr double kSymmetryTolerance = 1.0e-12;

    // Function checks whether the value is safely positive
    double clamp_positive(double value, double fallback)
    {
        return (value > kPositiveFloor) ? value : fallback; // condition: (value > kPositiveFloor), if
                                                            // true: return value else return fallback
    }

    bool matrix_has_square_storage(const std::vector<double>& a, int n)
    {
        if (n < 0)
        {
            return false;
        }

        return a.size() == static_cast<std::size_t>(n) * static_cast<std::size_t>(n);
    }

    std::vector<double> make_spd_matrix(std::size_t n, const MatrixGenerationOptions &options)
    {
        std::vector<double> c(n * n, 0.0); // This is the storage for the matrix (row-major format)
        std::vector<double> row_abs_sums(n, 0.0);

        std::mt19937_64 rng(options.seed); // Random number generator
        const double amplitude =
            clamp_positive(options.amplitude,
                           1.0); // amplitude to control the range of the off-diagonal matrix entries
        const double delta = clamp_positive(options.nugget, kDefaultAdder);
        std::uniform_real_distribution<double> offdiag_dist(
            -amplitude, amplitude); // build the probability distribution

        for (std::size_t i = 0; i < n; ++i)
        {
            const std::size_t i_row = i * n;

            for (int j = 0; j < i; ++j)
            {
                const double value =
                    offdiag_dist(rng); // generate random number in the range [-amplitude, amplitude]

                // Lower triangle element (and its symmetric one)
                c[i_row + j] = value;
                c[j * n + i] = value;

                // Compute absolute value and add to sum - for the SPD condition check later
                const double abs_value = std::fabs(value);
                row_abs_sums[i] += abs_value;
                row_abs_sums[j] += abs_value;
            }
        }

        for (std::size_t i = 0; i < n; ++i)
        {
            // Compute and store the diagonal element - we do this as row_abs_sums[i] + delta to ensure
            // it is large enough for SPD
            c[i * n + i] = row_abs_sums[i] + delta;
        }

        return c;
    }

    double coursework_brief_corr(double x, double y, double s)
    {
        const double diff = x - y;
        return 0.99 * std::exp(-0.5 * 16.0 * diff * diff / (s * s));
    }
} // namespace

bool matrix_is_strictly_diagonally_dominant(const std::vector<double>& a, int n)
{
    if (!matrix_has_square_storage(a, n))
    {
        return false;
    }

    for (int i = 0; i < n; ++i)
    {
        double offdiag_abs_sum = 0.0;

        for (int j = 0; j < n; ++j)
        {
            if (i == j)
            {
                continue;
            }

            offdiag_abs_sum += std::fabs(
                a[static_cast<std::size_t>(i) * static_cast<std::size_t>(n) +
                  static_cast<std::size_t>(j)]);
        }

        const double diag =
            a[static_cast<std::size_t>(i) * static_cast<std::size_t>(n) + static_cast<std::size_t>(i)];
        if (!(diag > offdiag_abs_sum))
        {
            return false;
        }
    }

    return true;
}

bool matrix_satisfies_generated_spd_conditions(const std::vector<double>& a, int n)
{
    if (!matrix_has_square_storage(a, n))
    {
        return false;
    }

    for (int i = 0; i < n; ++i)
    {
        const double diag =
            a[static_cast<std::size_t>(i) * static_cast<std::size_t>(n) + static_cast<std::size_t>(i)];
        if (diag <= 0.0)
        {
            return false;
        }

        for (int j = i + 1; j < n; ++j)
        {
            const double upper =
                a[static_cast<std::size_t>(i) * static_cast<std::size_t>(n) +
                  static_cast<std::size_t>(j)];
            const double lower =
                a[static_cast<std::size_t>(j) * static_cast<std::size_t>(n) +
                  static_cast<std::size_t>(i)];

            if (std::fabs(upper - lower) > kSymmetryTolerance)
            {
                return false;
            }
        }
    }

    return matrix_is_strictly_diagonally_dominant(a, n);
}

std::vector<double> make_coursework_brief_matrix(int n)
{
    if (n <= 0)
    {
        return {};
    }

    const std::size_t size = static_cast<std::size_t>(n);
    std::vector<double> c(size * size, 0.0);
    const double s = static_cast<double>(n);

    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            c[static_cast<std::size_t>(i) * size + static_cast<std::size_t>(j)] =
                coursework_brief_corr(static_cast<double>(i), static_cast<double>(j), s);
        }

        c[static_cast<std::size_t>(i) * size + static_cast<std::size_t>(i)] = 1.0;
    }

    return c;
}

std::vector<double> make_gershgorin_adjusted_copy(const std::vector<double>& a, int n)
{
    if (!matrix_has_square_storage(a, n))
    {
        return {};
    }

    std::vector<double> adjusted = a;

    for (int i = 0; i < n; ++i)
    {
        double offdiag_abs_sum = 0.0;

        for (int j = 0; j < n; ++j)
        {
            if (i == j)
            {
                continue;
            }

            offdiag_abs_sum += std::fabs(
                adjusted[static_cast<std::size_t>(i) * static_cast<std::size_t>(n) +
                         static_cast<std::size_t>(j)]);
        }

        const std::size_t diag_index =
            static_cast<std::size_t>(i) * static_cast<std::size_t>(n) + static_cast<std::size_t>(i);
        const double required_diag = offdiag_abs_sum + kDefaultAdder;
        if (adjusted[diag_index] < required_diag)
        {
            adjusted[diag_index] = required_diag;
        }
    }

    return adjusted;
}

std::vector<double> make_generated_spd_matrix(int n)
{
    const MatrixGenerationOptions options;
    return make_generated_spd_matrix(n, options);
}

std::vector<double> make_generated_spd_matrix(int n, const MatrixGenerationOptions &options)
{
    if (n <= 0)
    {
        return {}; // check matrix size is valid
    }

    const std::vector<double> generated = make_spd_matrix(static_cast<std::size_t>(n), options);

#if defined(MPHIL_DEBUG_CHECKS) && MPHIL_DEBUG_CHECKS
    assert(matrix_satisfies_generated_spd_conditions(generated, n));
#endif

    return generated;
}
