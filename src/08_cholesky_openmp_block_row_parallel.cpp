/**
 * @file 08_cholesky_openmp_block_row_parallel.cpp
 * @brief Blocked OpenMP Cholesky implementation with parallel trailing updates.
 */

/*
Simple blocked structure:

1. Factorise the diagonal block in the upper triangle (serial)
2. Compute the block row to the right (serial)
3. Update the trailing upper-triangular tiles in parallel
*/

#include "cholesky_helpers.h"
#include "cholesky_versions.h"

#include <algorithm>
#include <cmath>
#include <cstddef>

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
     * @brief Factor the active diagonal block of the upper triangle.
     *
     * @param c Row-major matrix storage updated in place.
     * @param n Matrix dimension.
     * @param k Active block start index.
     * @param kend Active block end index.
     */
    void factor_diagonal_block(double *c,
                               std::size_t n,
                               std::size_t k,
                               std::size_t kend)
    {
        for (std::size_t p = k; p < kend; ++p)
        {
            double *row_p = c + p * n;
            const double diag = std::sqrt(row_p[p]);
            row_p[p] = diag;
            const double inv_diag = 1.0 / diag;

            for (std::size_t j = p + 1; j < kend; ++j)
            {
                row_p[j] *= inv_diag;
            }

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
    }

    /**
     * @brief Solve the block row to the right of the active diagonal block.
     *
     * @param c Row-major matrix storage updated in place.
     * @param n Matrix dimension.
     * @param k Active block start index.
     * @param kend Active block end index.
     */
    void solve_panel_block_row(double *c,
                               std::size_t n,
                               std::size_t k,
                               std::size_t kend)
    {
        for (std::size_t p = k; p < kend; ++p)
        {
            double *row_p = c + p * n;
            const double inv_diag = 1.0 / row_p[p];

            for (std::size_t j = kend; j < n; ++j)
            {
                double sum = row_p[j];

                for (std::size_t s = k; s < p; ++s)
                {
                    const double *row_s = c + s * n;
                    sum -= row_s[p] * row_s[j];
                }

                row_p[j] = sum * inv_diag;
            }
        }
    }

    /**
     * @brief Update one diagonal trailing tile in the upper triangle.
     *
     * @param c Row-major matrix storage updated in place.
     * @param n Matrix dimension.
     * @param ii Tile start row.
     * @param iend Tile end row.
     * @param k Active block start index.
     * @param kend Active block end index.
     */
    void update_diagonal_trailing_tile_upper(double *c,
                                             std::size_t n,
                                             std::size_t ii,
                                             std::size_t iend,
                                             std::size_t k,
                                             std::size_t kend)
    {
        for (std::size_t p = k; p < kend; ++p)
        {
            const double *row_p = c + p * n;

            for (std::size_t i = ii; i < iend; ++i)
            {
                double *row_i = c + i * n;
                const double cpi = row_p[i];

                for (std::size_t j = i; j < iend; ++j)
                {
                    row_i[j] -= cpi * row_p[j];
                }
            }
        }
    }

    /**
     * @brief Update one off-diagonal trailing tile in the upper triangle.
     *
     * @param c Row-major matrix storage updated in place.
     * @param n Matrix dimension.
     * @param ii Tile start row.
     * @param iend Tile end row.
     * @param jj Tile start column.
     * @param jend Tile end column.
     * @param k Active block start index.
     * @param kend Active block end index.
     */
    void update_off_diagonal_trailing_tile_upper(double *c,
                                                 std::size_t n,
                                                 std::size_t ii,
                                                 std::size_t iend,
                                                 std::size_t jj,
                                                 std::size_t jend,
                                                 std::size_t k,
                                                 std::size_t kend)
    {
        for (std::size_t p = k; p < kend; ++p)
        {
            const double *row_p = c + p * n;

            for (std::size_t i = ii; i < iend; ++i)
            {
                double *row_i = c + i * n;
                const double cpi = row_p[i];

                for (std::size_t j = jj; j < jend; ++j)
                {
                    row_i[j] -= cpi * row_p[j];
                }
            }
        }
    }

    /**
     * @brief Update all tiles in one trailing block row of the upper triangle.
     *
     * @param c Row-major matrix storage updated in place.
     * @param n Matrix dimension.
     * @param block_size Tile size.
     * @param ii Block-row start index.
     * @param iend Block-row end index.
     * @param k Active block start index.
     * @param kend Active block end index.
     */
    void update_trailing_block_row_upper(double *c,
                                         std::size_t n,
                                         std::size_t block_size,
                                         std::size_t ii,
                                         std::size_t iend,
                                         std::size_t k,
                                         std::size_t kend)
    {
        update_diagonal_trailing_tile_upper(c, n, ii, iend, k, kend);

        for (std::size_t jj = ii + block_size; jj < n; jj += block_size)
        {
            const std::size_t jend = min_sz(jj + block_size, n);
            update_off_diagonal_trailing_tile_upper(c, n, ii, iend, jj, jend, k, kend);
        }
    }
} // namespace

void cholesky_openmp_block_row_parallel(double *c, std::size_t n, std::size_t block_size)
{
    if (c == nullptr || n == 0 || block_size == 0)
    {
        return;
    }

    for (std::size_t k = 0; k < n; k += block_size)
    {
        const std::size_t kend = min_sz(k + block_size, n);

        factor_diagonal_block(c, n, k, kend);
        solve_panel_block_row(c, n, k, kend);

        const std::size_t block_rows =
            (n > kend) ? ((n - kend + block_size - 1) / block_size) : 0;

        if (block_rows < 2)
        {
            for (std::size_t row_index = 0; row_index < block_rows; ++row_index)
            {
                const std::size_t ii = kend + row_index * block_size;
                const std::size_t iend = min_sz(ii + block_size, n);
                update_trailing_block_row_upper(c, n, block_size, ii, iend, k, kend);
            }
        }
        else
        {
            // Use a guided schedule for block rows because early rows own more trailing tiles
            // than later ones, and guided chunk shrinking balances that uneven workload.
#pragma omp parallel for schedule(guided)
            for (std::ptrdiff_t row_index = 0;
                 row_index < static_cast<std::ptrdiff_t>(block_rows);
                 ++row_index)
            {
                const std::size_t ii = kend + static_cast<std::size_t>(row_index) * block_size;
                const std::size_t iend = min_sz(ii + block_size, n);
                update_trailing_block_row_upper(c, n, block_size, ii, iend, k, kend);
            }
        }
    }

    // Mirror the updated upper triangle into the lower half so this blocked OpenMP path
    // preserves the library's full symmetric output contract.
    cholesky_detail::mirror_upper_to_lower(c, n);
}
