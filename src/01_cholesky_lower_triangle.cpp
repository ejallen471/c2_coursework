/**
 * @file 01_cholesky_lower_triangle.cpp
 * @brief Single-threaded lower-triangular Cholesky factorisation mirrored into full storage.
 */

/*
In this implementation we exploit symmetry and factorise a single triangle only.

Here we compute the lower triangle in place and reflect it into the upper half
once the factorisation is complete.
*/

#include "cholesky_helpers.h"
#include "cholesky_versions.h"

#include <cmath>

void cholesky_lower_triangle(double *c, const std::size_t n)
{
    /*
    Performs an in-place Cholesky decomposition of an n x n matrix.

    Parameters
    ----------
    c : double*
        Pointer to the first element of a contiguous array storing the
        matrix in row-major order. The matrix must be symmetric positive
        definite.

    n : std::size_t
        Dimension of the matrix, assuming a square matrix.
    */

    // Loop over the pivots (diagonals).
    for (std::size_t p = 0; p < n; ++p)
    {
        double *row_p = c + p * n;

        // Read the current diagonal element, take the square root and store in place.
        const double diag = std::sqrt(row_p[p]);
        row_p[p] = diag;

        // Scale the active column below the pivot.
        for (std::size_t i = p + 1; i < n; ++i)
        {
            double *row_i = c + i * n;
            row_i[p] /= diag;
        }

        // Update only the lower triangle of the trailing submatrix.
        for (std::size_t i = p + 1; i < n; ++i)
        {
            double *row_i = c + i * n;
            const double cip = row_i[p];

            for (std::size_t j = p + 1; j <= i; ++j)
            {
                row_i[j] -= cip * c[j * n + p];
            }
        }
    }

    cholesky_detail::mirror_lower_to_upper(c, static_cast<int>(n));
}
