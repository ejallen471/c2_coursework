/*
Here we parallelise without cache blocking.

At a high level, our steps are
1. build one pivot row (serial)
2. finish that row (serial) - scale by reciprocal of pivot
3. use it to update the rest of the matrix (parallel) - outer product update

Single-thread improvements used here:
- cache pointer to pivot row
- replace repeated division by multiplication with reciprocal
- avoid parallelising the light row-scaling step
- use std::size_t consistently
*/

#include "cholesky_helpers.h"
#include "cholesky_versions.h"

#include <cmath>
#include <cstddef>

void cholesky_openmp_1(double *c, std::size_t n)
{
    for (std::size_t p = 0; p < n; ++p)
    {
        double *row_p = c + p * n;
        const std::size_t start = p + 1;

        // Compute pivot
        const double diag = std::sqrt(row_p[p]);
        row_p[p] = diag;

        // Scale the rest of the pivot row using one reciprocal
        const double inv_diag = 1.0 / diag;
        for (std::size_t j = start; j < n; ++j)
        {
            row_p[j] *= inv_diag;
        }

        // Update trailing upper triangle in parallel
#pragma omp parallel for schedule(static)
        for (std::ptrdiff_t j = static_cast<std::ptrdiff_t>(start);
             j < static_cast<std::ptrdiff_t>(n);
             ++j)
        {
            const std::size_t j_index = static_cast<std::size_t>(j);
            double *row_j = c + j_index * n;
            const double cpj = row_p[j_index];

            for (std::size_t i = j_index; i < n; ++i)
            {
                row_j[i] -= cpj * row_p[i];
            }
        }
    }

    cholesky_detail::mirror_upper_to_lower(c, static_cast<int>(n));
}