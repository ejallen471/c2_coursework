/*
Parallelise with cache blocking

At a high level, our steps are
1. build several pivot rows (serial)
2. finish those rows (serial)
3. use them to update the rest of the matrix (parallel)

*/

#include "cholesky_helpers.h"
#include "cholesky_versions.h"

#include <algorithm>
#include <cmath>
#include <cstddef>

void cholesky_openmp_3(double *c, std::size_t n, std::size_t block_size)
{

    // Move across the matrix block by block.
    for (std::size_t k = 0; k < n; k += block_size)
    {
        // End index of the current block.
        const std::size_t kend = std::min(k + block_size, n);

        /*

        PART 1 — Factorise the diagonal block

        This is just a small Cholesky decomposition performed on
        the current block.

        Example if B = 3:

        Before:

        | A00 A01 A02 |
        | A01 A11 A12 |
        | A02 A12 A22 |

        After:

        | r00 r01 r02 |
        |  0  r11 r12 |
        |  0   0  r22 |

        This part is kept SERIAL because:
        - the block is small
        - pivots depend on earlier pivots
        - parallel overhead would dominate
        */

        for (std::size_t p = k; p < kend; ++p)
        {
            // Pointer to row p
            double *row_p = c + p * n;

            // Compute the diagonal element
            const double diag = std::sqrt(row_p[p]);
            row_p[p] = diag;

            const double inv_diag = 1.0 / diag;

            // Scale the remainder of the row within the block
            for (std::size_t j = p + 1; j < kend; ++j)
            {
                row_p[j] *= inv_diag;
            }

            // Update the rest of the block using this pivot row
            for (std::size_t j = p + 1; j < kend; ++j)
            {
                double *row_j = c + j * n;
                const double cpj = row_p[j];

                for (std::size_t i = j; i < kend; ++i)
                {
                    row_j[i] -= cpj * row_p[i];
                }
            }
        }

        /*
        PART 2 — Compute the block row to the right

        We now extend the rows of the factorised block across the
        remaining columns of the matrix.

        This computes entries:

        rows k..kend-1
        columns kend..n-1

        After this step the rows of the block are fully complete.

        Example:

        | r00 r01 r02 | r03 r04 r05 |
        |  0  r11 r12 | r13 r14 r15 |
        |  0   0  r22 | r23 r24 r25 |

        This version keeps this step SERIAL.
        */

        for (std::size_t p = k; p < kend; ++p)
        {
            const double diag = c[p * n + p];

            for (std::size_t j = kend; j < n; ++j)
            {
                double sum = c[p * n + j];

                // Subtract contributions from earlier rows in the block
                for (std::size_t s = k; s < p; ++s)
                {
                    sum -= c[s * n + p] * c[s * n + j];
                }

                // Finish the computation of this element
                c[p * n + j] = sum / diag;
            }
        }

        /*
        PART 3 — Update the trailing matrix

        Now we update the bottom-right region:

        A22 ← A22 − R12ᵀ R12

        This uses the completed block rows to update the rest of the
        matrix.

        This is the MOST EXPENSIVE part of the algorithm.

        Each iteration of j updates one destination row of A22.

        Because each thread works on different rows, the work is safe
        to parallelise.
        */

#pragma omp parallel for schedule(static)
        for (std::size_t j = kend; j < n; ++j)
        {
            // Pointer to row j of the matrix
            double *row_j = c + j * n;

            for (std::size_t i = j; i < n; ++i)
            {
                double sum = 0.0;

                // Accumulate contributions from each row in the block
                for (std::size_t p = k; p < kend; ++p)
                {
                    sum += c[p * n + j] * c[p * n + i];
                }

                row_j[i] -= sum;
            }
        }
    }

    // Copy the computed upper triangle into the lower triangle
    cholesky_detail::mirror_upper_to_lower(c, static_cast<int>(n));
}
