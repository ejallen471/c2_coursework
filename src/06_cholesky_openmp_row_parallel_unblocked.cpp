/**
 * @file 06_cholesky_openmp_row_parallel_unblocked.cpp
 * @brief OpenMP Cholesky implementation with parallel trailing-row updates.
 */

/*
Here we parallelise without cache blocking.

At a high level, our steps are
1. build one pivot row (serial)
2. finish that row (serial) - scale by reciprocal of pivot
3. use it to update the rest of the matrix (parallel) - outer product update - row by row

We also include Single-thread improvements from earlier iterations
*/

#include "cholesky_helpers.h"
#include "cholesky_versions.h"

#include <cmath>
#include <cstddef>

void cholesky_openmp_row_parallel_unblocked(double *c, std::size_t n)
{
    // Create the worker team once for the whole factorisation so we avoid paying thread
    // start-up costs on every pivot and can rely on OpenMP's implicit barriers between phases.
#pragma omp parallel
    {
        for (std::size_t p = 0; p < n; ++p)
        {
            double *row_p = c + p * n;
            const std::size_t start = p + 1;
            const std::size_t tail = n - start;

            // Keep pivot extraction and row scaling on a single thread because every worker
            // must see a fully normalised pivot row before the trailing update can begin.
#pragma omp single
            {
                const double diag = std::sqrt(row_p[p]);
                row_p[p] = diag;

                const double inv_diag = 1.0 / diag;
                for (std::size_t j = start; j < n; ++j)
                {
                    row_p[j] *= inv_diag;
                }
            }

            if (tail > 128)
            {
                // Use dynamic scheduling for long tails because the amount of work per row
                // shrinks as `j` increases, so fixed assignment would leave stragglers.
#pragma omp for schedule(dynamic, 2)
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
            else
            {
                // Use static scheduling for short tails because the work is already small and
                // the cheaper scheduler avoids dynamic-queue overhead dominating the update.
#pragma omp for schedule(static)
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
        }
    }

    // Mirror the computed upper triangle back into the lower half so this OpenMP kernel
    // returns the same full-matrix storage contract as the rest of the library.
    cholesky_detail::mirror_upper_to_lower(c, static_cast<int>(n));
}
