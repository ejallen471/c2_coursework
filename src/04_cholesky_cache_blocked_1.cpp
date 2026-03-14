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

                std::size_t j = jj;
                for (; j + 3 < jend; j += 4)
                {
                    row_p[j + 0] -= coeff * row_s[j + 0];
                    row_p[j + 1] -= coeff * row_s[j + 1];
                    row_p[j + 2] -= coeff * row_s[j + 2];
                    row_p[j + 3] -= coeff * row_s[j + 3];
                }

                for (; j < jend; ++j)
                {
                    row_p[j] -= coeff * row_s[j];
                }
            }

            const double inv_diag = 1.0 / row_p[p];

            std::size_t j = jj;
            for (; j + 3 < jend; j += 4)
            {
                row_p[j + 0] *= inv_diag;
                row_p[j + 1] *= inv_diag;
                row_p[j + 2] *= inv_diag;
                row_p[j + 3] *= inv_diag;
            }

            for (; j < jend; ++j)
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
            std::size_t j = i;

            for (; j + 3 < iend; j += 4)
            {
                double sum0 = 0.0;
                double sum1 = 0.0;
                double sum2 = 0.0;
                double sum3 = 0.0;

                for (std::size_t p = k; p < kend; ++p)
                {
                    const double *row_p = c + p * n;
                    const double upi = row_p[i];

                    sum0 += upi * row_p[j + 0];
                    sum1 += upi * row_p[j + 1];
                    sum2 += upi * row_p[j + 2];
                    sum3 += upi * row_p[j + 3];
                }

                row_i[j + 0] -= sum0;
                row_i[j + 1] -= sum1;
                row_i[j + 2] -= sum2;
                row_i[j + 3] -= sum3;
            }

            for (; j < iend; ++j)
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
            std::size_t j = jj;

            for (; j + 3 < jend; j += 4)
            {
                double sum0 = 0.0;
                double sum1 = 0.0;
                double sum2 = 0.0;
                double sum3 = 0.0;

                for (std::size_t p = k; p < kend; ++p)
                {
                    const double *row_p = c + p * n;
                    const double upi = row_p[i];

                    sum0 += upi * row_p[j + 0];
                    sum1 += upi * row_p[j + 1];
                    sum2 += upi * row_p[j + 2];
                    sum3 += upi * row_p[j + 3];
                }

                row_i[j + 0] -= sum0;
                row_i[j + 1] -= sum1;
                row_i[j + 2] -= sum2;
                row_i[j + 3] -= sum3;
            }

            for (; j < jend; ++j)
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
