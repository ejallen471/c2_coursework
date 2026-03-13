#ifndef MATRIX_H
#define MATRIX_H

#include <cstdint>
#include <vector>

struct MatrixGenerationOptions
{
    std::uint64_t seed = 20260310ULL;
    double amplitude = 1.0;
    double nugget = 1.0e-3;
};

std::vector<double> make_generated_spd_matrix(int n);
std::vector<double> make_generated_spd_matrix(int n, const MatrixGenerationOptions& options);

#endif
