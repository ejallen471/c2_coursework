/**
 * @file matrix.cpp
 * @brief Functions for building and checking symmetric positive definite matrices.
 */

#include "matrix.h"

#include <cassert>
#include <cmath>
#include <cstddef>
#include <random>

namespace
{
// Small value added to the diagonal so the generated matrix is safely SPD.
constexpr double kDefaultAdder = 1.0e-3;

// Used to decide whether an input option is meaningfully positive.
constexpr double kPositiveFloor = 1.0e-12;

// Allowed tolerance when checking whether a matrix is symmetric.
constexpr double kSymmetryTolerance = 1.0e-12;

/**
 * @brief Replace a non-positive option with a safe fallback.
 *
 * @param value User-supplied value.
 * @param fallback Value to use if `value` is not safely positive.
 * @return A positive value that is safe to use.
 */
double clamp_positive(double value, double fallback)
{
    return (value > kPositiveFloor) ? value : fallback;
}

/**
 * @brief Check whether a vector has the right size for an `n x n` matrix.
 *
 * @param a Matrix storage in row-major order.
 * @param n Matrix size.
 * @return `true` if the vector size matches `n * n`.
 */
bool matrix_has_square_storage(const std::vector<double>& a, int n)
{
    if (n < 0)
    {
        return false;
    }

    return a.size() == static_cast<std::size_t>(n) * static_cast<std::size_t>(n);
}

/**
 * @brief Build a random dense SPD matrix.
 *
 * The matrix is made symmetric first. Then each diagonal entry is set large enough
 * to make the matrix strictly diagonally dominant, which gives an SPD matrix here.
 *
 * @param n Matrix size.
 * @param options Generator settings such as seed and diagonal slack.
 * @return Random SPD matrix in row-major order.
 */
std::vector<double> make_spd_matrix(std::size_t n, const MatrixGenerationOptions& options)
{
    std::vector<double> c(n * n, 0.0);
    std::vector<double> row_abs_sums(n, 0.0);

    std::mt19937_64 rng(options.seed);

    const double amplitude = clamp_positive(options.amplitude, 1.0);
    const double delta = clamp_positive(options.nugget, kDefaultAdder);

    std::uniform_real_distribution<double> offdiag_dist(-amplitude, amplitude);

    // Fill the off-diagonal entries and mirror them so the matrix is symmetric.
    for (std::size_t i = 0; i < n; ++i)
    {
        const std::size_t i_row = i * n;

        for (std::size_t j = 0; j < i; ++j)
        {
            const double value = offdiag_dist(rng);

            c[i_row + j] = value;
            c[j * n + i] = value;

            const double abs_value = std::fabs(value);
            row_abs_sums[i] += abs_value;
            row_abs_sums[j] += abs_value;
        }
    }

    // Make each diagonal entry larger than the sum of the off-diagonal magnitudes
    // in its row. This makes the matrix strictly diagonally dominant.
    for (std::size_t i = 0; i < n; ++i)
    {
        c[i * n + i] = row_abs_sums[i] + delta;
    }

    return c;
}

/**
 * @brief Correlation function used for the coursework brief matrix.
 *
 * @param x First sample position.
 * @param y Second sample position.
 * @param s Global scale factor.
 * @return Correlation value.
 */
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

    // For each row, check that the diagonal entry is larger than the sum of the
    // absolute values of all off-diagonal entries in that row.
    for (int i = 0; i < n; ++i)
    {
        double offdiag_abs_sum = 0.0;

        for (int j = 0; j < n; ++j)
        {
            if (i == j)
            {
                continue;
            }

            offdiag_abs_sum +=
                std::fabs(a[static_cast<std::size_t>(i) * static_cast<std::size_t>(n) +
                            static_cast<std::size_t>(j)]);
        }

        const double diag = a[static_cast<std::size_t>(i) * static_cast<std::size_t>(n) +
                              static_cast<std::size_t>(i)];

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

    // Check positive diagonal and symmetry first.
    for (int i = 0; i < n; ++i)
    {
        const double diag = a[static_cast<std::size_t>(i) * static_cast<std::size_t>(n) +
                              static_cast<std::size_t>(i)];

        if (diag <= 0.0)
        {
            return false;
        }

        for (int j = i + 1; j < n; ++j)
        {
            const double upper = a[static_cast<std::size_t>(i) * static_cast<std::size_t>(n) +
                                   static_cast<std::size_t>(j)];

            const double lower = a[static_cast<std::size_t>(j) * static_cast<std::size_t>(n) +
                                   static_cast<std::size_t>(i)];

            if (std::fabs(upper - lower) > kSymmetryTolerance)
            {
                return false;
            }
        }
    }

    // Then check diagonal dominance.
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

    // Fill the matrix using the coursework brief correlation function.
    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            c[static_cast<std::size_t>(i) * size + static_cast<std::size_t>(j)] =
                coursework_brief_corr(static_cast<double>(i), static_cast<double>(j), s);
        }

        // Force the diagonal to exactly 1.0.
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

    // Increase the diagonal where needed so every row becomes strictly diagonally dominant.
    for (int i = 0; i < n; ++i)
    {
        double offdiag_abs_sum = 0.0;

        for (int j = 0; j < n; ++j)
        {
            if (i == j)
            {
                continue;
            }

            offdiag_abs_sum +=
                std::fabs(adjusted[static_cast<std::size_t>(i) * static_cast<std::size_t>(n) +
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

std::vector<double> make_generated_spd_matrix(int n, const MatrixGenerationOptions& options)
{
    if (n <= 0)
    {
        // Invalid size -> return an empty matrix.
        return {};
    }

    const std::vector<double> generated = make_spd_matrix(static_cast<std::size_t>(n), options);

#if defined(DEBUG_CHECKS) && DEBUG_CHECKS
    // In debug builds, verify that the generator really produced what we expected.
    assert(matrix_satisfies_generated_spd_conditions(generated, n));
#endif

    return generated;
}