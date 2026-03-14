/*
Parallelise with cache blocking

At a high level, our steps are
1. build several pivot rows (serial)
2. finish those rows (parallel in columns)
3. use them to update the rest of the matrix (parallel) - outer product updates

*/

#include "cholesky_helpers.h"
#include "cholesky_versions.h"

#include <algorithm>
#include <cmath>
#include <cstddef>

void cholesky_openmp_4(double *c, std::size_t n, std::size_t block_size)
{

    // Move through the matrix one diagonal block at a time.
    for (std::size_t k = 0; k < n; k += block_size)
    {
        // End of the current block - For the last block, kend may be smaller than k + B.
        const std::size_t kend = std::min(k + block_size, n);

        /*
        PART 1: Factorise the first diagonal block A

        This is ordinary unblocked Cholesky, but restricted to the
        current diagonal block only.

        After this part, the small diagonal block has been turned into
        the corresponding upper-triangular Cholesky factor.

        Example idea for a 3x3 diagonal block:

            before:
                [ A00 A01 A02 ]
                [ A01 A11 A12 ]
                [ A02 A12 A22 ]

            after:
                [ r00 r01 r02 ]
                [  0  r11 r12 ]
                [  0   0  r22 ]

        Only the upper triangle is used during the factorisation.
        */
        for (std::size_t p = k; p < kend; ++p)
        {
            // Pointer to row p.
            double *row_p = c + p * n;

            // Compute the diagonal entry of the factor.
            const double diag = std::sqrt(row_p[p]);
            row_p[p] = diag;

            // Reciprocal used to scale the rest of the row in the block.
            const double inv_diag = 1.0 / diag;

            // Scale the part of row p that lies inside the current block.
            // This completes row p within the diagonal block.
            for (std::size_t j = p + 1; j < kend; ++j)
            {
                row_p[j] *= inv_diag;
            }

            // Update the remaining rows of the block using row p.
            for (std::size_t j = p + 1; j < kend; ++j) // restrict to upper portion of the matrix
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
        PART 2: Compute the block row to the right

        The diagonal block is now factorised, but the rows in this block
        are not yet complete across the whole matrix.

        This step fills the entries to the right of the block, so that
        rows k..kend-1 become fully computed.

        For each row p in the block, and for each column j to the right,
        subtract contributions from earlier rows in the same block, then
        divide by the diagonal value.

        This version parallelises over j (columns to the right) for each
        fixed p.

        The iterations are independent because:
        - each thread writes to a different c[p*n + j]
        - all threads only read previously computed values
        */
        for (std::size_t p = k; p < kend; ++p)
        {
            const double diag = c[p * n + p];

#pragma omp parallel for schedule(static)
            for (std::size_t j = kend; j < n; ++j)
            {

                // Start from the current matrix value.
                double sum = c[p * n + j];

                // Subtract the contributions from earlier rows in this block.
                //
                // Once rows k..p-1 are known, they contribute to row p.
                for (std::size_t s = k; s < p; ++s)
                {
                    sum -= c[s * n + p] * c[s * n + j];
                }

                // Divide by the diagonal entry of row p to complete c[p, j].
                c[p * n + j] = sum / diag;
            }
        }

        /*
        PART 3: Update the trailing matrix

        Now that the current block rows are complete, use them to update
        the bottom-right trailing matrix.

        In block notation:
            A22 <- A22 - A12^T * A12

        Each iteration of j updates one destination row of the trailing
        upper triangle, so parallelising over j is safe:
        - different threads write to different rows
        - all threads read the already computed block row A12

        Work here is restricted to the upper triangle by starting i at j.
        */
#pragma omp parallel for schedule(static)
        for (std::size_t j = kend; j < n; ++j)
        {

            // Pointer to destination row j.
            double *row_j = c + j * n;

            // Update the upper-triangular part of row j.
            for (std::size_t i = j; i < n; ++i)
            {
                double sum = 0.0;

                // Accumulate the contribution from every row p in the
                // completed block k..kend-1.
                for (std::size_t p = k; p < kend; ++p)
                {
                    sum += c[p * n + j] * c[p * n + i];
                }

                // Apply the block update.
                row_j[i] -= sum;
            }
        }
    }

    // We computed the upper triangle only, thus now have to reflect
    cholesky_detail::mirror_upper_to_lower(c, n);
}
