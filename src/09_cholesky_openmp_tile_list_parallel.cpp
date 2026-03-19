/**
 * @file 09_cholesky_openmp_tile_list_parallel.cpp
 * @brief Blocked OpenMP Cholesky implementation driven by explicit tile work lists.
 */

#include "cholesky_helpers.h"
#include "cholesky_versions.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

namespace
{
    /**
     * @brief Describe one blocked tile range for work-list execution.
     *
     * @note The indices refer to row-major matrix storage.
     */
    struct TileRange
    {
        std::size_t row_begin;
        std::size_t row_end;
        std::size_t col_begin;
        std::size_t col_end;
        bool diagonal;
    };

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

                // Ask OpenMP to vectorise the contiguous inner-tile update because each `i`
                // iteration writes a distinct element and therefore has no loop-carried hazard.
#pragma omp simd
                for (std::size_t i = j; i < kend; ++i)
                {
                    row_j[i] -= cpj * row_p[i];
                }
            }
        }
    }

    /**
     * @brief Solve one panel tile to the right of the active diagonal block.
     *
     * @param c Row-major matrix storage updated in place.
     * @param n Matrix dimension.
     * @param k Active block start index.
     * @param kend Active block end index.
     * @param jj Tile start column.
     * @param jend Tile end column.
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
            const double inv_diag = 1.0 / row_p[p];

            for (std::size_t j = jj; j < jend; ++j)
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
     * @param k Active block start index.
     * @param kend Active block end index.
     * @param ii Tile start row.
     * @param iend Tile end row.
     */
    void update_diagonal_tile(double *c,
                              std::size_t n,
                              std::size_t k,
                              std::size_t kend,
                              std::size_t ii,
                              std::size_t iend)
    {
        for (std::size_t p = k; p < kend; ++p)
        {
            const double *row_p = c + p * n;

            for (std::size_t i = ii; i < iend; ++i)
            {
                double *row_i = c + i * n;
                const double cpi = row_p[i];

                // Ask OpenMP to vectorise the diagonal tile update so each lane handles an
                // independent output entry within the triangular tile.
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
     * @param k Active block start index.
     * @param kend Active block end index.
     * @param ii Tile start row.
     * @param iend Tile end row.
     * @param jj Tile start column.
     * @param jend Tile end column.
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
        for (std::size_t p = k; p < kend; ++p)
        {
            const double *row_p = c + p * n;

            for (std::size_t i = ii; i < iend; ++i)
            {
                double *row_i = c + i * n;
                const double cpi = row_p[i];

                // Ask OpenMP to vectorise the off-diagonal tile update because the writes are
                // contiguous and independent once the panel contribution is fixed.
#pragma omp simd
                for (std::size_t j = jj; j < jend; ++j)
                {
                    row_i[j] -= cpi * row_p[j];
                }
            }
        }
    }
} // namespace

void cholesky_openmp_tile_list_parallel(double *c, std::size_t n, std::size_t block_size)
{
    if (c == nullptr || n == 0 || block_size == 0)
    {
        return;
    }

    const std::size_t max_trailing_tiles =
        (n + block_size - 1) / block_size;

    std::vector<TileRange> panel_tiles;
    std::vector<TileRange> trailing_tiles;

    panel_tiles.reserve(max_trailing_tiles);
    trailing_tiles.reserve(max_trailing_tiles * (max_trailing_tiles + 1) / 2);

    // Create the worker team once so the serial work-list construction can feed a stable pool
    // of threads during every blocked iteration instead of rebuilding teams repeatedly.
#pragma omp parallel
    {
        for (std::size_t k = 0; k < n; k += block_size)
        {
            const std::size_t kend = std::min(k + block_size, n);

            // Use one thread to build deterministic work lists because every worker must agree
            // on the same tile descriptors before the later worksharing loop consumes them.
#pragma omp single
            {
                factor_diagonal_block(c, n, k, kend);

                panel_tiles.clear();
                for (std::size_t jj = kend; jj < n; jj += block_size)
                {
                    panel_tiles.push_back({k, kend, jj, std::min(jj + block_size, n), false});
                }

                // Keep the panel solve serial: usually too small to justify threading.
                for (const TileRange &tile : panel_tiles)
                {
                    solve_panel_tile(c, n, k, kend, tile.col_begin, tile.col_end);
                }

                trailing_tiles.clear();
                for (std::size_t ii = kend; ii < n; ii += block_size)
                {
                    const std::size_t iend = std::min(ii + block_size, n);
                    trailing_tiles.push_back({ii, iend, ii, iend, true});

                    for (std::size_t jj = ii + block_size; jj < n; jj += block_size)
                    {
                        trailing_tiles.push_back({ii, iend, jj, std::min(jj + block_size, n), false});
                    }
                }
            }

            if (trailing_tiles.size() < 4)
            {
                // Keep very small tile lists serial because worksharing overhead would outweigh
                // the benefit of parallelism when only a handful of tiles remain.
#pragma omp single
                {
                    for (const TileRange &tile : trailing_tiles)
                    {
                        if (tile.diagonal)
                        {
                            update_diagonal_tile(c, n, k, kend, tile.row_begin, tile.row_end);
                        }
                        else
                        {
                            update_offdiagonal_tile(
                                c, n, k, kend,
                                tile.row_begin, tile.row_end,
                                tile.col_begin, tile.col_end);
                        }
                    }
                }
            }
            else
            {
                // Distribute one tile descriptor at a time dynamically because diagonal and edge
                // tiles cost more than interior ones, so a static split would load-balance badly.
#pragma omp for schedule(dynamic, 1)
                for (std::ptrdiff_t tile_index = 0;
                     tile_index < static_cast<std::ptrdiff_t>(trailing_tiles.size());
                     ++tile_index)
                {
                    const TileRange &tile = trailing_tiles[static_cast<std::size_t>(tile_index)];

                    if (tile.diagonal)
                    {
                        update_diagonal_tile(c, n, k, kend, tile.row_begin, tile.row_end);
                    }
                    else
                    {
                        update_offdiagonal_tile(
                            c, n, k, kend,
                            tile.row_begin, tile.row_end,
                            tile.col_begin, tile.col_end);
                    }
                }
            }
        }
    }

    // Mirror the upper-triangular result into the lower half so all implementations return
    // the same symmetric matrix layout to benchmark and correctness code.
    cholesky_detail::mirror_upper_to_lower(c, n);
}
