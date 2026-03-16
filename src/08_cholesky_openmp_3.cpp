/**
 * @file 08_cholesky_openmp_3.cpp
 * @brief Blocked OpenMP Cholesky implementation with dynamic tile scheduling.
 */

/*
Here we use blocks throughout, we still have the same following steps

1. Factorise the diagonal block (serial)
2. Compute teh block row to the right (serial)
3. Update the trailing matrix using that block row - this is done in parallel
*/

#include "cholesky_helpers.h"
#include "cholesky_versions.h"

#include <algorithm>
#include <cmath>
#include <cstddef>

void cholesky_openmp_3(double *c, std::size_t n, std::size_t block_size)
{

    // Move across the matrix block by block.

    /*
    For each block we do three things
        1. Factorise the diagonal block
        2. Compute teh block row to the right
        3. Update the trailing matrix using that block row - this is done in parallel
    */

    for (std::size_t k = 0; k < n; k += block_size)
    {
        const std::size_t kend = std::min(k + block_size, n); // Get the end idx of the block

        /*
        1. Factorise the diagonal block
        */

        // Loop through the rows of the current diagonal block - one row at a time
        for (std::size_t p = k; p < kend; ++p)
        {
            double *row_p = c + p * n;

            const double diag = std::sqrt(row_p[p]);
            row_p[p] = diag;
            const double inv_diag = 1.0 / diag;

            // Update each elemt to the right to be divided by the sqrt of the diagonal
            for (std::size_t j = p + 1; j < kend; ++j)
            {
                row_p[j] *= inv_diag;
            }

            // Update the remaining submatrix within the diagonal block
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
        2. Compute the block row to the right
        */

        // Loop through the rows of the current block
        for (std::size_t p = k; p < kend; ++p)
        {
            double *row_p = c + p * n;
            const double inv_diag = 1.0 / row_p[p];

            for (std::size_t j = kend; j < n; ++j)
            {
                double sum = row_p[j];

                // Subtract the outer product contributions for all but the first row
                for (std::size_t s = k; s < p; ++s)
                {
                    const double *row_s = c + s * n;
                    sum -= row_s[p] * row_s[j];
                }

                row_p[j] = sum * inv_diag;
            }
        }

        /*
        3. — Update the trailing matrix in parallel over trailing blocks
        */

        // calculate and store the number of tiles.
        // Condition: (n > kend) if true ((n - kend + block_size - 1) / block_size) else 0
        const std::size_t tiles = (n > kend) ? ((n - kend + block_size - 1) / block_size) : 0;
        const std::size_t num_upper_tiles = tiles * (tiles + 1) / 2;

#pragma omp parallel for schedule(dynamic, 1)
        /*
        OPENMP - Here we split the loop iterations across multiple threads.

        We use dynamic scheduling because the work per iteration is not uniform.
        As the factorisation moves down the matrix, the remaining upper-triangular
        region gets smaller, so later iterations often have less work than earlier ones.

        schedule(dynamic, 1) means iterations are assigned to threads one at a time
        as threads become free, which helps load balance when iteration costs vary.
        */

        for (std::ptrdiff_t tile_id = 0;
             tile_id < static_cast<std::ptrdiff_t>(num_upper_tiles);
             ++tile_id)
        {
            // Turns a single number like tile_id = 0,1,2 into upper triangular tile coordinates
            std::size_t id = static_cast<std::size_t>(tile_id);
            std::size_t ti = 0;
            std::size_t row_len = tiles;

            while (id >= row_len)
            {
                id -= row_len;
                ++ti;
                --row_len;
            }

            const std::size_t tj = ti + id;

            // Convert tile coordinates into matrix indices
            const std::size_t ii = kend + ti * block_size;
            const std::size_t jj = kend + tj * block_size;

            const std::size_t iend = std::min(ii + block_size, n);
            const std::size_t jend = std::min(jj + block_size, n);

            // Case 1: diagonal tile
            if (ii == jj)
            {
                // Diagonal tile: update upper triangle only
                for (std::size_t i = ii; i < iend; ++i)
                {
                    double *row_i = c + i * n;

                    for (std::size_t j = i; j < jend; ++j)
                    {
                        double sum = 0.0;

                        for (std::size_t p = k; p < kend; ++p)
                        {
                            const double *row_p = c + p * n;
                            sum += row_p[i] * row_p[j];
                        }

                        row_i[j] -= sum;
                    }
                }
            }
            else
            {
                // Off-diagonal tile: update full tile
                for (std::size_t i = ii; i < iend; ++i)
                {
                    double *row_i = c + i * n;

                    for (std::size_t j = jj; j < jend; ++j)
                    {
                        double sum = 0.0;

                        for (std::size_t p = k; p < kend; ++p)
                        {
                            const double *row_p = c + p * n;
                            sum += row_p[i] * row_p[j];
                        }

                        row_i[j] -= sum;
                    }
                }
            }
        }
    }

    cholesky_detail::mirror_upper_to_lower(c, n);
}
