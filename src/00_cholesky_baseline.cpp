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
        double *row_p = c + p * n;

        // the diagonal entry becomes the square root of the current pivot value
        const double diag = std::sqrt(row_p[p]);
        row_p[p] = diag;

        // Loop across the column to the right of the diagonal
        for (std::size_t j = p + 1; j < n; ++j)
        {
            // update each element by dividing by the pivot element square rooted
            row_p[j] /= diag;
        }

        // Loop down the rows below the diagonal in column p
        for (std::size_t i = p + 1; i < n; ++i)
        {
            double *row_i = c + i * n;
            row_i[p] /= diag;
        }

        // loop over the submatrix (created by getting rid of the column and row of the pivot)
        for (std::size_t j = p + 1; j < n; ++j)
        {
            const double cpj = row_p[j];

            for (std::size_t k = p + 1; k < n; ++k)
            {
                double *row_k = c + k * n;
                row_k[j] -= row_k[p] * cpj; // update according to the update rule
            }
        }
    }
}
