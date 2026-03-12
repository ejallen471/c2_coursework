/*
This file contains the single-threaded Cholesky implementations called from mphil_dis_cholesky.cpp.

These implementations include

1. Baseline
2. Lower triangular Only
2. Loop Cleanup
3. Access Pattern Aware
4. Cache Blocked
5. Vectorisation Friendly

*/

#include "cholesky_versions.h"
#include "cholesky_helpers.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <vector>

// BASELINE IMPLEMENTATION FROM COURSEWORK DOCUMENT
void cholesky_baseline(double *c, const std::size_t n)
{

    for (std::size_t p = 0; p < n; p++)
    { /* move along the diagonal of the matrix */
        const std::size_t p_row = p * n;
        const double diag = std::sqrt(c[p_row + p]);
        c[p_row + static_cast<std::size_t>(p)] = diag;

        for (std::size_t j = p + 1; j < n; j++)
        { /* update row to right of diagonal element */
            c[p_row + j] /= diag;
        }

        for (std::size_t i = p + 1; i < n; i++)
        { /* update column below diagonal element */
            const std::size_t i_row = i * n;
            c[i_row + p] /= diag;
        }

        for (std::size_t j = p + 1; j < n; j++)
        { /* update submatrix below-right of diagonal element */

            for (std::size_t k = p + 1; k < n; k++)
            {
                const std::size_t k_row = k * n;
                c[k_row + j] -= c[k_row + p] * c[p_row + j];
            }
        }
    }
}
