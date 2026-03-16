/**
 * @file 00_cholesky_baseline.cpp
 * @brief Baseline dense Cholesky factorisation that updates the full matrix in place.
 */

#include "cholesky_versions.h"
#include <cmath>

void cholesky_baseline(double *c, const std::size_t n) // note in baseline n is type int, to be consistent with my other methods and calling i have changed to std::size_t
{
    // Outer loop over pivot position
    for (std::size_t p = 0; p < n; ++p)
    {
        // the diagonal entry becomes the square root of the current pivot value
        const double diag = std::sqrt(c[p * n + p]);
        c[p * n + p] = diag;

        // Loop across the column to the right of the diagonal
        for (std::size_t j = p + 1; j < n; ++j)
        {
            // update each element by dividing by the pivot element square rooted
            c[p * n + j] /= diag;
        }

        // Loop down the rows below the diagonal in column p
        for (std::size_t i = p + 1; i < n; ++i)
        {
            c[i * n + p] /= diag;
        }

        // loop over the submatrix (created by getting rid of the column and row of the pivot)
        for (std::size_t j = p + 1; j < n; ++j)
        {
            for (std::size_t k = p + 1; k < n; ++k)
            {
                c[k * n + j] -= c[k * n + p] * c[n * p + j]; // update according to the update rule
            }
        }
    }
}
