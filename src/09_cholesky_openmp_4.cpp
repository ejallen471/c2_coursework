/**
 * @file 09_cholesky_openmp_4.cpp
 * @brief Blocked OpenMP Cholesky implementation driven by explicit tile work lists.
 */

/*
Parallelise with cache blocking with added optimisations

1. Factorise the diagonal block (serial)
2. Compute teh block row to the right (serial)
3. Update the trailing matrix using that block row - this is done in parallel
*/

#include "cholesky_helpers.h"
#include "cholesky_versions.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

namespace // anonymous namespace - everything inside is only visible in this source file
{

    // store the tile indices in one struct
    struct TileRange
    {
        std::size_t row_begin;
        std::size_t row_end;
        std::size_t col_begin;
        std::size_t col_end;
        bool diagonal;
    };

    // Function to factorise the current diagonal block
    void factor_diagonal_block(double *c, std::size_t n, std::size_t k, std::size_t kend)
    {
        // Loop through the rows of the diagonal block one by one
        for (std::size_t p = k; p < kend; ++p)
        {
            // Point to the start of row p
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

    // Compute the tile of the panel to the right of the current diagonal block
    void solve_panel_tile(double *c, std::size_t n, std::size_t k, std::size_t kend,
                          std::size_t jj, std::size_t jend)
    {
        // Loop through rows of the diagonal block
        for (std::size_t p = k; p < kend; ++p)
        {
            double *row_p = c + p * n;
            const double inv_diag = 1.0 / row_p[p];

            for (std::size_t j = jj; j < jend; ++j)
            {
                double sum = row_p[j];

                // Subtract contributions from the outer product
                for (std::size_t s = k; s < p; ++s)
                {
                    const double *row_s = c + s * n;
                    sum -= row_s[p] * row_s[j];
                }

                row_p[j] = sum * inv_diag;
            }
        }
    }

    // Update one diagonal tile in the trailing submatrix
    void update_diagonal_tile(double *c, std::size_t n, std::size_t k, std::size_t kend,
                              std::size_t ii, std::size_t iend)
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

    // Update the off-diagonal tiles in the trailing submatrix
    void update_offdiagonal_tile(double *c, std::size_t n, std::size_t k, std::size_t kend,
                                 std::size_t ii, std::size_t iend,
                                 std::size_t jj, std::size_t jend)
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
} // end of namespace

void cholesky_openmp_4(double *c, std::size_t n, std::size_t block_size)
{
    // Precompute the maximum tile count
    const std::size_t max_trailing_tiles =
        (n == 0) ? 0 : ((n + block_size - 1) / block_size);

    std::vector<TileRange> panel_tiles;
    std::vector<TileRange> trailing_tiles;

    // Reserve memory so the vectors do not keep reallocating
    panel_tiles.reserve(max_trailing_tiles);
    trailing_tiles.reserve(max_trailing_tiles * (max_trailing_tiles + 1) / 2);

    // Move block by block along the diagonal
    for (std::size_t k = 0; k < n; k += block_size)
    {
        const std::size_t kend = std::min(k + block_size, n);

        factor_diagonal_block(c, n, k, kend);

        panel_tiles.clear();
        for (std::size_t jj = kend; jj < n; jj += block_size)
        {
            panel_tiles.push_back({k, kend, jj, std::min(jj + block_size, n), false});
        }

        // Sovle the panel, if the tile is less than two, the overhead from parallelising is not worth it
        if (panel_tiles.size() < 2)
        {
            for (const TileRange &tile : panel_tiles)
            {
                solve_panel_tile(c, n, k, kend, tile.col_begin, tile.col_end);
            }
        }
        else
        {
#pragma omp parallel for schedule(static)
            /*
            Parallelise the panel-tile solves.

            Static scheduling is used because these iterations have similar cost, so
            threads can be given fixed chunks with low scheduling overhead.
            */
            // Loop over tile indices
            for (std::ptrdiff_t tile_index = 0;
                 tile_index < static_cast<std::ptrdiff_t>(panel_tiles.size());
                 ++tile_index)
            {
                // Each thread gets one panel tile and solves it
                const TileRange &tile = panel_tiles[static_cast<std::size_t>(tile_index)];
                solve_panel_tile(c, n, k, kend, tile.col_begin, tile.col_end);
            }
        }

        trailing_tiles.clear(); // Clear the old trailing tiles

        // For each tile row in the trailing matrix
        for (std::size_t ii = kend; ii < n; ii += block_size)
        {
            // First add its diagonal tile
            const std::size_t iend = std::min(ii + block_size, n);
            trailing_tiles.push_back({ii, iend, ii, iend, true});

            // Then add the off-diagonal tiles to the right of that diagonal tile
            for (std::size_t jj = ii + block_size; jj < n; jj += block_size)
            {
                trailing_tiles.push_back({ii, iend, jj, std::min(jj + block_size, n), false});
            }
        }

        // Update the trailing tiles
        if (trailing_tiles.size() < 4) // if very few, do serially
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
                        c, n, k, kend, tile.row_begin, tile.row_end, tile.col_begin, tile.col_end);
                }
            }
        }
        else
        {

#pragma omp parallel for schedule(guided)
            /*
            Parallelise the trailing tile updates across threads.

            Here, we have guided scheduling because the tile updates are not all the same cost.

            In guided scheduling, OpenMP starts by giving threads larger chunks of work,
            then gradually reduces the chunk size as the loop progresses. T
            */

            // parallel loop over trailing tile list
            for (std::ptrdiff_t tile_index = 0;
                 tile_index < static_cast<std::ptrdiff_t>(trailing_tiles.size());
                 ++tile_index)
            {
                const TileRange &tile = trailing_tiles[static_cast<std::size_t>(tile_index)];

                // Update upper triangle only if it is a diagonal trailing tile
                if (tile.diagonal)
                {
                    update_diagonal_tile(c, n, k, kend, tile.row_begin, tile.row_end);
                }
                else // update the whole off-diagonal tile
                {
                    update_offdiagonal_tile(
                        c, n, k, kend, tile.row_begin, tile.row_end, tile.col_begin, tile.col_end);
                }
            }
        }
    }

    cholesky_detail::mirror_upper_to_lower(c, n);
}
