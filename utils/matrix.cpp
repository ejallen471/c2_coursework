/*
This. code builds a dense symmetric positive definite (SPD) matrix.

This is stored as a flat std::vector<double> and built using covariance kernels from 1D points

At a high level we

1. Pick n points on the line
2. measure how similar each pair o points is using a kernel
3. put that similarity into matrix entry (i,j)
4. add a small value to the diagonal to make the matrix numerically safer

*/

#include "matrix.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <random>

namespace
{
// Default generator settings chosen to keep the covariance matrix well-conditioned.
constexpr double kDefaultLengthScale = 0.2;
constexpr double kDefaultNugget = 1.0e-3;
constexpr double kPositiveFloor = 1.0e-12;

// Replace non-positive parameters with a safe fallback before they are used in kernel formulas.
double clamp_positive(double value, double fallback)
{
    return (value > kPositiveFloor) ? value : fallback;
}

// Exponential covariance kernel k(x, y) = exp(-|x-y| / ell).
double exponential_kernel(double x, double y, double length_scale)
{
    const double safe_length_scale = clamp_positive(length_scale, kDefaultLengthScale);
    return std::exp(-std::fabs(x - y) / safe_length_scale);
}

// Build the 1D input locations used by the covariance kernels.
std::vector<double> make_points(int n, const MatrixGenerationOptions& options, std::mt19937_64& rng)
{
    std::vector<double> points(static_cast<std::size_t>(n), 0.0);

    // Either sample points randomly in [0, 1] or place them deterministically on a uniform grid.
    if (options.randomize_points)
    {
        std::uniform_real_distribution<double> point_dist(0.0, 1.0);
        for (double& point : points)
        {
            point = point_dist(rng);
        }
    }
    else if (n == 1)
    {
        points[0] = 0.0;
    }
    else
    {
        const double denom = static_cast<double>(n - 1);
        for (int i = 0; i < n; ++i)
        {
            points[static_cast<std::size_t>(i)] = static_cast<double>(i) / denom;
        }
    }

    return points;
}

// Dispatch the non-mixture kernels through one helper so the main generator loop stays simple.
double kernel_value(CovarianceKernel kernel, double x, double y, double length_scale)
{
    switch (kernel)
    {
    case CovarianceKernel::SquaredExponential:
        return corr(x, y, length_scale);

    case CovarianceKernel::Exponential:
        return exponential_kernel(x, y, length_scale);

    case CovarianceKernel::RandomMixture:
        break;
    }

    return 0.0;
}
} // namespace

double corr(double x, double y, double length_scale)
{
    // Squared-exponential covariance kernel k(x, y) = exp(-0.5 * ((x-y)/ell)^2).
    const double safe_length_scale = clamp_positive(length_scale, kDefaultLengthScale);
    const double diff = (x - y) / safe_length_scale;
    return std::exp(-0.5 * diff * diff);
}

std::vector<double> make_generated_spd_matrix(int n)
{
    return make_generated_spd_matrix(n, MatrixGenerationOptions{});
}

std::vector<double> make_generated_spd_matrix(int n, const MatrixGenerationOptions& options)
{
    if (n <= 0)
    {
        return {};
    }

    // Store the dense matrix in one row-major vector with entry (i, j) at c[i * n + j].
    const std::size_t n_size = static_cast<std::size_t>(n);
    std::vector<double> c(n_size * n_size, 0.0);

    // Seed the generator once so matrix generation is repeatable for a given options.seed value.
    std::mt19937_64 rng(options.seed);
    const std::vector<double> points = make_points(n, options, rng);

    // Amplitude scales the whole covariance matrix, while the nugget stabilises the diagonal.
    const double amplitude = clamp_positive(options.amplitude, 1.0);
    const double nugget = clamp_positive(options.nugget, kDefaultNugget);

    double se_length_scale = clamp_positive(options.length_scale, kDefaultLengthScale);
    double exp_length_scale = se_length_scale;

    double se_weight = 1.0;
    double exp_weight = 0.0;

    // For the random-mixture option, draw two kernel length scales and normalised mixture weights.
    if (options.kernel == CovarianceKernel::RandomMixture)
    {
        std::uniform_real_distribution<double> length_scale_dist(0.12, 0.35);
        std::uniform_real_distribution<double> weight_dist(0.2, 1.0);

        se_length_scale = length_scale_dist(rng);
        exp_length_scale = length_scale_dist(rng);

        se_weight = weight_dist(rng);
        exp_weight = weight_dist(rng);

        const double weight_sum = se_weight + exp_weight;
        se_weight /= weight_sum;
        exp_weight /= weight_sum;
    }

    // Fill only the lower triangle and mirror into the upper triangle to preserve symmetry exactly.
    for (int i = 0; i < n; ++i)
    {
        const std::size_t row_offset = static_cast<std::size_t>(i) * n_size;

        for (int j = 0; j <= i; ++j)
        {
            const double x = points[static_cast<std::size_t>(i)];
            const double y = points[static_cast<std::size_t>(j)];

            double value = 0.0;

            // The random-mixture path blends squared-exponential and exponential kernels.
            if (options.kernel == CovarianceKernel::RandomMixture)
            {
                value = se_weight * corr(x, y, se_length_scale) +
                    exp_weight * exponential_kernel(x, y, exp_length_scale);
            }
            else
            {
                value = kernel_value(options.kernel, x, y, se_length_scale);
            }

            value *= amplitude;

            if (i == j)
            {
                // Add a small diagonal nugget so the matrix is numerically safer for Cholesky.
                value += nugget;
            }

            c[row_offset + static_cast<std::size_t>(j)] = value;
            c[static_cast<std::size_t>(j) * n_size + static_cast<std::size_t>(i)] = value;
        }
    }

    return c;
}
