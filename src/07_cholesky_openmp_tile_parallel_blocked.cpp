/**
 * @file 07_cholesky_openmp_tile_parallel_blocked.cpp
 * @brief Blocked OpenMP Cholesky with tile-parallel trailing update.
 *
 * Strategy:
 * 1. Factor the diagonal block serially
 * 2. Solve the panel block row serially
 * 3. Update trailing upper-triangular tiles in parallel
 */

#include "cholesky_helpers.h"
#include "cholesky_versions.h"

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

                // Ask OpenMP to vectorise the contiguous upper-triangular tile update because
                // each `j` iteration is independent and SIMD reduces scalar update overhead.
#pragma omp simd
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

                // Ask OpenMP to vectorise the off-diagonal tile update for the same reason:
                // each output entry is independent once `cpi` and `row_p` are fixed.
#pragma omp simd
                for (std::size_t j = jj; j < jend; ++j)
                {
                    row_i[j] -= cpi * row_p[j];
                }
            }
        }
    }

    /**
     * @brief Update one trailing tile, dispatching to the diagonal or off-diagonal kernel.
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
    void update_trailing_tile_upper(double *c,
                                    std::size_t n,
                                    std::size_t ii,
                                    std::size_t iend,
                                    std::size_t jj,
                                    std::size_t jend,
                                    std::size_t k,
                                    std::size_t kend)
    {
        if (ii == jj)
        {
            update_diagonal_trailing_tile_upper(c, n, ii, iend, k, kend);
        }
        else
        {
            update_off_diagonal_trailing_tile_upper(c, n, ii, iend, jj, jend, k, kend);
        }
    }
} // namespace

void cholesky_openmp_tile_parallel_blocked(double *c, std::size_t n, std::size_t block_size)
{
    if (c == nullptr || n == 0 || block_size == 0)
    {
        return;
    }

    // Create one team for the full blocked factorisation so workers stay alive across block
    // iterations and only the expensive trailing-tile phase is shared out repeatedly.
#pragma omp parallel
    {
        for (std::size_t k = 0; k < n; k += block_size)
        {
            const std::size_t kend = min_sz(k + block_size, n);

            // Keep the panel factorisation serial because the diagonal block and its panel solve
            // are strongly sequential and usually too small for extra threading to pay off.
#pragma omp single
            {
                factor_diagonal_block(c, n, k, kend);
                solve_panel_block_row(c, n, k, kend);
            }

            const std::size_t trailing_blocks =
                (n > kend) ? ((n - kend + block_size - 1) / block_size) : 0;

            if (trailing_blocks < 2)
            {
                // For tiny trailing regions, keep one thread in charge because the work queue
                // setup and barrier cost would exceed any benefit from parallelising.
#pragma omp single
                {
                    for (std::size_t ii = kend; ii < n; ii += block_size)
                    {
                        const std::size_t iend = min_sz(ii + block_size, n);

                        for (std::size_t jj = ii; jj < n; jj += block_size)
                        {
                            const std::size_t jend = min_sz(jj + block_size, n);
                            update_trailing_tile_upper(c, n, ii, iend, jj, jend, k, kend);
                        }
                    }
                }
            }
            else
            {
                // For larger trailing regions, distribute one tile-row per chunk dynamically so
                // threads can steal work when diagonal and edge tiles cost different amounts.
#pragma omp for schedule(dynamic, 1)
                for (std::ptrdiff_t ii = static_cast<std::ptrdiff_t>(kend);
                     ii < static_cast<std::ptrdiff_t>(n);
                     ii += static_cast<std::ptrdiff_t>(block_size))
                {
                    const std::size_t i0 = static_cast<std::size_t>(ii);
                    const std::size_t iend = min_sz(i0 + block_size, n);

                    for (std::size_t jj = i0; jj < n; jj += block_size)
                    {
                        const std::size_t jend = min_sz(jj + block_size, n);
                        update_trailing_tile_upper(c, n, i0, iend, jj, jend, k, kend);
                    }
                }
            }
        }
    }

    // Mirror the completed upper triangle so callers see the same full symmetric storage as
    // the single-threaded kernels and the correctness checks can compare like with like.
    cholesky_detail::mirror_upper_to_lower(c, n);
}
