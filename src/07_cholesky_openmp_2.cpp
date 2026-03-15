/*
Here we parallelise without cache blocking.

At a high level, our steps are
1. build one pivot row (serial)
2. finish that row (parallel) - scale by reciprocal of pivot
3. use it to update the rest of the matrix (parallel) - outer product update

Single-thread improvements used here:
- cache pointer to pivot row
- replace repeated division by multiplication with reciprocal
- simplify indexing
*/

#include "cholesky_helpers.h"
#include "cholesky_versions.h"

#include <cmath>
#include <cstddef>

void cholesky_openmp_2(double *c, std::size_t n)
{
    for (std::size_t p = 0; p < n; ++p)
    {
        double *row_p = c + p * n;
        const std::size_t start = p + 1;

        // Compute the pivot
        const double diag = std::sqrt(row_p[p]);
        row_p[p] = diag;

        // Use one reciprocal rather than repeated division
        const double inv_diag = 1.0 / diag;

#pragma omp parallel for schedule(static)
        /*
        OPENMP: Divide work across threads

        Each iteration scales one independent entry of the pivot row by
        the same reciprocal, thus the work per iteaction is essentially the same therefore static
        scheduling, to minimise overhead

        */
        for (std::ptrdiff_t j = static_cast<std::ptrdiff_t>(start);
             j < static_cast<std::ptrdiff_t>(n);
             ++j)
        {
            row_p[j] *= inv_diag;
        }

#pragma omp parallel for schedule(static)
        /*
        Each iteration updates one row of the remaining upper triangle using the
        completed pivot row.

        Static scheduling reduces the overhead and the actual work does not change much moving downwards
        */
        for (std::ptrdiff_t j = static_cast<std::ptrdiff_t>(start);
             j < static_cast<std::ptrdiff_t>(n);
             ++j)
        {
            double *row_j = c + j * n;
            const double cpj = row_p[j];

            for (std::size_t i = j; i < n; ++i)
            {
                row_j[i] -= cpj * row_p[i];
            }
        }
    }

    cholesky_detail::mirror_upper_to_lower(c, static_cast<int>(n));
}
