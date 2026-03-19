/**
 * @file 05_cholesky_blocked_tile_kernels_unrolled.cpp
 * @brief Single-threaded blocked Cholesky implementation with unrolled tile kernels.
 */

/*
In this implementation we perform a cache-blocked Cholesky decomposition with explicit unrolling.

The matrix is processed in diagonal blocks. For each block we:
1. Factorise the diagonal block
2. Compute the block row to the right
3. Update the trailing matrix using the block we just computed

The matrix is stored in row-major order in a flat array.
*/

#include "cholesky_helpers.h"
#include "cholesky_versions.h"

#include <algorithm>
#include <cmath>
#include <vector>

namespace
{
    /**
     * @brief Return the smaller of two tile bounds.
     *
     * @param a First bound.
     * @param b Second bound.
     * @return Smaller bound.
     */
    inline std::size_t min_sz(std::size_t a, std::size_t b)
    {
        return (a < b) ? a : b;
    }

    /**
     * @brief Factor one diagonal block and cache reciprocal pivots.
     *
     * @param c Row-major matrix storage updated in place.
     * @param n Matrix dimension.
     * @param k Active block start index.
     * @param block_end Active block end index.
     * @param diag_recip Scratch array for diagonal reciprocals.
     *
     * @note The input matrix is assumed to be symmetric positive definite.
     */
    void factor_diagonal_block(double *c,
                               std::size_t n,
                               std::size_t k,
                               std::size_t block_end,
                               double *diag_recip)
    {
        for (std::size_t p = k; p < block_end; ++p)
        {
            double *row_p = c + p * n;
            const std::size_t start = p + 1;
            const double diag = std::sqrt(row_p[p]);
            row_p[p] = diag;

            const double inv_diag = 1.0 / diag;
            diag_recip[p - k] = inv_diag;

            std::size_t j = start;
            for (; j + 3 < block_end; j += 4)
            {
                row_p[j] *= inv_diag;
                row_p[j + 1] *= inv_diag;
                row_p[j + 2] *= inv_diag;
                row_p[j + 3] *= inv_diag;
            }

            for (; j < block_end; ++j)
            {
                row_p[j] *= inv_diag;
            }

            for (std::size_t j_block = start; j_block < block_end; ++j_block)
            {
                double *row_j = c + j_block * n;
                const double cache_term = row_p[j_block];

                std::size_t i = j_block;
                for (; i + 3 < block_end; i += 4)
                {
                    row_j[i] -= cache_term * row_p[i];
                    row_j[i + 1] -= cache_term * row_p[i + 1];
                    row_j[i + 2] -= cache_term * row_p[i + 2];
                    row_j[i + 3] -= cache_term * row_p[i + 3];
                }

                for (; i < block_end; ++i)
                {
                    row_j[i] -= cache_term * row_p[i];
                }
            }
        }
    }

    /**
     * @brief Solve the panel block row to the right of the active diagonal block.
     *
     * @param c Row-major matrix storage updated in place.
     * @param n Matrix dimension.
     * @param k Active block start index.
     * @param block_end Active block end index.
     * @param diag_recip Cached reciprocal pivots for the diagonal block.
     */
    void solve_panel_block_row(double *c,
                               std::size_t n,
                               std::size_t k,
                               std::size_t block_end,
                               const double *diag_recip)
    {
        for (std::size_t j = block_end; j < n; ++j)
        {
            for (std::size_t p = k; p < block_end; ++p)
            {
                double *row_p = c + p * n;
                double sum = row_p[j];
                std::size_t s = k;

                for (; s + 3 < p; s += 4)
                {
                    const double *row_s0 = c + s * n;
                    const double *row_s1 = c + (s + 1) * n;
                    const double *row_s2 = c + (s + 2) * n;
                    const double *row_s3 = c + (s + 3) * n;

                    sum -= row_s0[p] * row_s0[j];
                    sum -= row_s1[p] * row_s1[j];
                    sum -= row_s2[p] * row_s2[j];
                    sum -= row_s3[p] * row_s3[j];
                }

                for (; s < p; ++s)
                {
                    const double *row_s = c + s * n;
                    sum -= row_s[p] * row_s[j];
                }

                row_p[j] = sum * diag_recip[p - k];
            }
        }
    }

    /**
     * @brief Update the trailing matrix tiles using the active block panel.
     *
     * @param c Row-major matrix storage updated in place.
     * @param n Matrix dimension.
     * @param k Active block start index.
     * @param block_end Active block end index.
     * @param block_size Tile size.
     */
    void update_trailing_tiles(double *c,
                               std::size_t n,
                               std::size_t k,
                               std::size_t block_end,
                               std::size_t block_size)
    {
        for (std::size_t ii = block_end; ii < n; ii += block_size)
        {
            const std::size_t i_end = min_sz(ii + block_size, n);

            for (std::size_t jj = ii; jj < n; jj += block_size)
            {
                const std::size_t j_end = min_sz(jj + block_size, n);

                for (std::size_t i = ii; i < i_end; ++i)
                {
                    double *row_i = c + i * n;
                    std::size_t j = (jj == ii) ? i : jj;

                    for (; j + 3 < j_end; j += 4)
                    {
                        double sum0 = 0.0;
                        double sum1 = 0.0;
                        double sum2 = 0.0;
                        double sum3 = 0.0;

                        for (std::size_t p = k; p < block_end; ++p)
                        {
                            const double *row_p = c + p * n;
                            const double upi = row_p[i];

                            sum0 += upi * row_p[j];
                            sum1 += upi * row_p[j + 1];
                            sum2 += upi * row_p[j + 2];
                            sum3 += upi * row_p[j + 3];
                        }

                        row_i[j] -= sum0;
                        row_i[j + 1] -= sum1;
                        row_i[j + 2] -= sum2;
                        row_i[j + 3] -= sum3;
                    }

                    for (; j < j_end; ++j)
                    {
                        double sum = 0.0;

                        for (std::size_t p = k; p < block_end; ++p)
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
} // namespace

void cholesky_blocked_tile_kernels_unrolled(
    double *c, const std::size_t n, const std::size_t block_size)
{
    if (block_size == 0)
    {
        return;
    }

    std::vector<double> diag_recip(block_size, 0.0);

    for (std::size_t k = 0; k < n; k += block_size)
    {
        const std::size_t block_end = min_sz(k + block_size, n);

        factor_diagonal_block(c, n, k, block_end, diag_recip.data());
        solve_panel_block_row(c, n, k, block_end, diag_recip.data());
        update_trailing_tiles(c, n, k, block_end, block_size);
    }

    cholesky_detail::mirror_upper_to_lower(c, static_cast<int>(n));
}
