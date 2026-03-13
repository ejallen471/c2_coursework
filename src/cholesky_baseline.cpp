#include "cholesky_versions.h"

#include <cmath>

void cholesky_baseline(double* c, const std::size_t n)
{
    // Outer loop over pivot position
    for (std::size_t p = 0; p < n; ++p)
    {
        // Compute the starting index of row p in the flat array (row major -> its p * n)
        const std::size_t p_row = p * n;

        // the diagonal entry becomes the square root of the current pivot value
        const double diag = std::sqrt(c[p_row + p]);
        c[p_row + p] = diag;

        // Loop across the column to the right of the diagonal
        for (std::size_t j = p + 1; j < n; ++j)
        {
            c[p_row + j] /=
                diag; // update each element by dividing by the pivot element square rooted
        }

        // Loop down the rows below the diagonal in column p
        for (std::size_t i = p + 1; i < n; ++i)
        {
            const std::size_t i_row = i * n; // compute the starting index of row i
            c[i_row + p] /= diag;
        }

        // loop over the submatrix (created by getting rid of the column and row of the pivot)
        for (std::size_t j = p + 1; j < n; ++j)
        {
            for (std::size_t k = p + 1; k < n; ++k)
            {
                const std::size_t k_row = k * n;
                c[k_row + j] -= c[k_row + p] * c[p_row + j]; // update according to the update rule
            }
        }
    }
}
