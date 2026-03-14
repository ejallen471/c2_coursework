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

        // Finish the pivot row
#pragma omp parallel for schedule(static)
        for (std::ptrdiff_t j = static_cast<std::ptrdiff_t>(start);
             j < static_cast<std::ptrdiff_t>(n);
             ++j)
        {
            row_p[static_cast<std::size_t>(j)] *= inv_diag;
        }

        // Update the trailing upper triangle
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
