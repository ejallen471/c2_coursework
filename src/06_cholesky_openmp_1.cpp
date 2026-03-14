
/*
Here we parallelise without cache blocking

*/
#include "cholesky_helpers.h"
#include "cholesky_versions.h"

#include <cmath>
#include <cstddef>

void cholesky_openmp_1(double *c, std::size_t n)
{
    const int n_int = static_cast<int>(n);

    for (int p = 0; p < n_int; ++p)
    {
        const std::size_t p_index = static_cast<std::size_t>(p);
        const std::size_t p_row = p_index * n;
        const int start = p + 1;

        const double diag = std::sqrt(c[p_row + p_index]);
        c[p_row + p_index] = diag;

#pragma omp parallel for schedule(static)
        for (int j = start; j < n_int; ++j)
        {
            c[p_row + static_cast<std::size_t>(j)] /= diag;
        }

#pragma omp parallel for schedule(static)
        for (int j = start; j < n_int; ++j)
        {
            const std::size_t j_index = static_cast<std::size_t>(j);
            double *row_j = c + j_index * n;
            const double cpj = c[p_row + j_index];

            for (std::size_t i = j_index; i < n; ++i)
            {
                row_j[i] -= cpj * c[p_row + i];
            }
        }
    }

    cholesky_detail::mirror_upper_to_lower(c, n_int);
}
