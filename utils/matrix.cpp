/*
This code builds dense symmetric positive definite (SPD) matrices.

We generate a random symmetric matrix and then enforce strict diagonal dominance:

    a_ii > sum_{j != i} |a_ij|

For symmetric matrices, that is a sufficient condition for SPD.
*/

#include "matrix.h"

#include <cmath>
#include <cstddef>
#include <random>

namespace
{
constexpr double kDefaultAdder = 1.0e-3;   // Compile time constant to add to the diagonal for
                                           // numerical stability and ensure SPD condition
constexpr double kPositiveFloor = 1.0e-12; // Compile time constant, Used to decide whether a
                                           // parameter should be treated as positive or invalid.

// Function checks whether the value is safely positive
double clamp_positive(double value, double fallback)
{
    return (value > kPositiveFloor) ? value : fallback; // condition: (value > kPositiveFloor), if
                                                        // true: return value else return fallback
}

std::vector<double> make_spd_matrix(std::size_t n, const MatrixGenerationOptions& options)
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
} // namespace

std::vector<double> make_generated_spd_matrix(int n)
{
    const MatrixGenerationOptions options;
    return make_generated_spd_matrix(n, options);
}

std::vector<double> make_generated_spd_matrix(int n, const MatrixGenerationOptions& options)
{
    if (n <= 0)
    {
        return {}; // check matrix size is valid
    }

    return make_spd_matrix(n, options);
}
