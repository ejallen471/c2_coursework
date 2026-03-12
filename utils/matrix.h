#ifndef MATRIX_H
#define MATRIX_H

#include <cstdint>
#include <vector>

enum class CovarianceKernel
{
    SquaredExponential,
    Exponential,
    RandomMixture,
};

struct MatrixGenerationOptions
{
    std::uint64_t seed = 20260310ULL;
    CovarianceKernel kernel = CovarianceKernel::RandomMixture;
    double amplitude = 1.0;
    double length_scale = 0.2;
    double nugget = 1.0e-3;
    bool randomize_points = true;
    bool sort_points = false;
};

double corr(double x, double y, double length_scale);

std::vector<double> make_generated_spd_matrix(int n);
std::vector<double> make_generated_spd_matrix(int n, const MatrixGenerationOptions& options);

#endif
