/**
 * @file 04_cholesky_blocked_tile_kernels.cpp
 * @brief Single-threaded blocked Cholesky implementation built around explicit tile kernels.
 */

/*
In this implementation, we focus on cache blocking while factorising only one triangle.

The blocked algorithm is organised around explicit tile kernels:
1. factorise the diagonal tile
2. solve each panel tile to the right
3. update each trailing tile as one owned unit

This keeps the structure ready for future tile-level parallelism without relying on
external libraries.
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
     * @brief Factor one diagonal tile of the upper-triangular blocked matrix.
     *
     * @param c Row-major matrix storage updated in place.
     * @param n Matrix dimension.
     * @param k Tile start index.
     * @param kend Tile end index.
     *
     * @note The input matrix is assumed to be symmetric positive definite.
     */
    void factor_diagonal_tile(double *c, std::size_t n, std::size_t k, std::size_t kend)
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

            for (std::size_t i = p + 1; i < kend; ++i)
            {
                double *row_i = c + i * n;
                const double upi = row_p[i];

                for (std::size_t j = i; j < kend; ++j)
                {
                    row_i[j] -= upi * row_p[j];
                }
            }
        }
    }

    /**
     * @brief Solve one panel tile to the right of the active diagonal tile.
     *
     * @param c Row-major matrix storage updated in place.
     * @param n Matrix dimension.
     * @param k Active tile start index.
     * @param kend Active tile end index.
     * @param jj Panel tile start column.
     * @param jend Panel tile end column.
     */
    void solve_panel_tile(double *c,
                          std::size_t n,
                          std::size_t k,
                          std::size_t kend,
                          std::size_t jj,
                          std::size_t jend)
    {
        for (std::size_t p = k; p < kend; ++p)
        {
            double *row_p = c + p * n;

            for (std::size_t s = k; s < p; ++s)
            {
                const double *row_s = c + s * n;
                const double coeff = row_s[p];

                for (std::size_t j = jj; j < jend; ++j)
                {
                    row_p[j] -= coeff * row_s[j];
                }
            }

            const double inv_diag = 1.0 / row_p[p];

            for (std::size_t j = jj; j < jend; ++j)
            {
                row_p[j] *= inv_diag;
            }
        }
    }

    /**
     * @brief Update one diagonal tile in the trailing matrix.
     *
     * @param c Row-major matrix storage updated in place.
     * @param n Matrix dimension.
     * @param k Active tile start index.
     * @param kend Active tile end index.
     * @param ii Trailing tile start row.
     * @param iend Trailing tile end row.
     */
    void update_diagonal_tile(double *c,
                              std::size_t n,
                              std::size_t k,
                              std::size_t kend,
                              std::size_t ii,
                              std::size_t iend)
    {
        for (std::size_t i = ii; i < iend; ++i)
        {
            double *row_i = c + i * n;

            for (std::size_t j = i; j < iend; ++j)
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

    /**
     * @brief Update one off-diagonal tile in the trailing matrix.
     *
     * @param c Row-major matrix storage updated in place.
     * @param n Matrix dimension.
     * @param k Active tile start index.
     * @param kend Active tile end index.
     * @param ii Trailing tile start row.
     * @param iend Trailing tile end row.
     * @param jj Trailing tile start column.
     * @param jend Trailing tile end column.
     */
    void update_offdiagonal_tile(double *c,
                                 std::size_t n,
                                 std::size_t k,
                                 std::size_t kend,
                                 std::size_t ii,
                                 std::size_t iend,
                                 std::size_t jj,
                                 std::size_t jend)
    {
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

    /**
     * @brief Update all trailing tiles owned by the active blocked step.
     *
     * @param c Row-major matrix storage updated in place.
     * @param n Matrix dimension.
     * @param block_size Tile size.
     * @param k Active tile start index.
     * @param kend Active tile end index.
     */
    void update_trailing_tiles(double *c,
                               std::size_t n,
                               std::size_t block_size,
                               std::size_t k,
                               std::size_t kend)
    {
        for (std::size_t ii = kend; ii < n; ii += block_size)
        {
            const std::size_t iend = min_sz(ii + block_size, n);
            update_diagonal_tile(c, n, k, kend, ii, iend);

            for (std::size_t jj = ii + block_size; jj < n; jj += block_size)
            {
                const std::size_t jend = min_sz(jj + block_size, n);
                update_offdiagonal_tile(c, n, k, kend, ii, iend, jj, jend);
            }
        }
    }
} // namespace

void cholesky_blocked_tile_kernels(double *c, const std::size_t n, const std::size_t block_size)
{
    if (block_size == 0)
    {
        return;
    }

    for (std::size_t k = 0; k < n; k += block_size)
    {
        const std::size_t kend = min_sz(k + block_size, n);

        factor_diagonal_tile(c, n, k, kend);

        for (std::size_t jj = kend; jj < n; jj += block_size)
        {
            const std::size_t jend = min_sz(jj + block_size, n);
            solve_panel_tile(c, n, k, kend, jj, jend);
        }

        update_trailing_tiles(c, n, block_size, k, kend);
    }

    cholesky_detail::mirror_upper_to_lower(c, static_cast<int>(n));
}
