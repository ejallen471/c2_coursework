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
} // namespace

void cholesky_cache_blocked_1(double *c, const std::size_t n, const std::size_t block_size)
{
    if (block_size == 0)
    {
        return;
    }

    for (std::size_t k = 0; k < n; k += block_size)
    {
        const std::size_t kend = std::min(k + block_size, n);

        factor_diagonal_tile(c, n, k, kend);

        for (std::size_t jj = kend; jj < n; jj += block_size)
        {
            const std::size_t jend = std::min(jj + block_size, n);
            solve_panel_tile(c, n, k, kend, jj, jend);
        }

        for (std::size_t ii = kend; ii < n; ii += block_size)
        {
            const std::size_t iend = std::min(ii + block_size, n);
            update_diagonal_tile(c, n, k, kend, ii, iend);

            for (std::size_t jj = ii + block_size; jj < n; jj += block_size)
            {
                const std::size_t jend = std::min(jj + block_size, n);
                update_offdiagonal_tile(c, n, k, kend, ii, iend, jj, jend);
            }
        }
    }

    cholesky_detail::mirror_upper_to_lower(c, static_cast<int>(n));
}