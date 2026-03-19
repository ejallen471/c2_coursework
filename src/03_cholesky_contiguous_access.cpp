/**
 * @file 03_cholesky_contiguous_access.cpp
 * @brief Single-threaded Cholesky variant organised for contiguous row-major updates.
 */

/*
In this implementation, we focus on memory locality and cache behaviour.

The standard Cholesky structure (which we keep) is

1. compute the diagonal element
2. scale the row to the right of the pivot
3. update the trailing submatrix using a rank-1 update (subtract the outer product)

Step three is the most expensive part. So we do the following:

- Update the trailing matrix row-by-row to ensure
that the inner loop writes to contiguous memory

- Keep the active row hot in cache so the source values are also
read contiguously.

- Reduce repeated index arithmetic by using row pointers.
This avoids recalculating expressions like i*n + j inside the inner loops
and simplifies the hot loop body.

*/

#include "cholesky_helpers.h"
#include "cholesky_versions.h"

#include <cmath>
#include <vector>

void cholesky_contiguous_access(double *c, const std::size_t n)
{
    // Reuse one scratch row so the active pivot row is cached outside the hot update loop.
    std::vector<double> cached_pivot_row(n, 0.0);
    double *const cached_pivot = cached_pivot_row.data();

    for (std::size_t p = 0; p < n; ++p)
    {
        double *row_p = c + p * n;
        const std::size_t start = p + 1;

        const double diag = std::sqrt(row_p[p]);
        row_p[p] = diag;
        const double diag_recip = 1.0 / diag;

        // Scale row p in place so both reads and writes stay contiguous.
        for (std::size_t j = start; j < n; ++j)
        {
            row_p[j] *= diag_recip;
        }

        for (std::size_t j = start; j < n; ++j)
        {
            cached_pivot[j] = row_p[j];
        }

        // Update trailing upper triangle row-by-row.
        for (std::size_t j = start; j < n; ++j)
        {
            double *row_j = c + j * n + j;
            const double *pivot_values = cached_pivot + j;
            const double cpj = cached_pivot[j];

            for (std::size_t len = n - j; len != 0; --len)
            {
                *row_j -= cpj * (*pivot_values);
                ++row_j;
                ++pivot_values;
            }
        }
    }

    cholesky_detail::mirror_upper_to_lower(c, n);
}
