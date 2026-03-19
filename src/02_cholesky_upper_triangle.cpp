/**
 * @file 02_cholesky_upper_triangle.cpp
 * @brief Single-threaded upper-triangular Cholesky factorisation mirrored into full storage.
 */

/*
In this implementation we exploit symmetry and factorise a single triangle only.

To keep the hot loop aligned with row-major storage, we compute the upper triangle
and reflect it into the lower half once the factorisation is complete.
*/

#include "cholesky_helpers.h"
#include "cholesky_versions.h"

#include <cmath>

void cholesky_upper_triangle(double *c, const std::size_t n)
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
        Dimension of the matrix - assuming square matrix
    */

    // Loop over the pivots (diagonals)
    for (std::size_t p = 0; p < n; ++p)
    {
        double *row_p = c + p * n;

        // Read the current diagonal element, take the square root and store in placee
        const double diag = std::sqrt(row_p[p]);
        row_p[p] = diag;

        // Scale the active row to the right of the pivot.
        for (std::size_t j = p + 1; j < n; ++j)
        {
            row_p[j] /= diag;
        }

        // Update only the upper triangle of the trailing submatrix.
        for (std::size_t j = p + 1; j < n; ++j)
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
