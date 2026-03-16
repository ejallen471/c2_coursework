/**
 * @file 05_cholesky_cache_blocked_2.cpp
 * @brief Second cache-blocked single-threaded Cholesky implementation with loop unrolling.
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

void cholesky_cache_blocked_2(double *c, const std::size_t n, const std::size_t block_size)
{
    if (block_size == 0)
    {
        return;
    }

    std::vector<double> diag_recip(block_size, 0.0);

    for (std::size_t k = 0; k < n; k += block_size)
    {
        const std::size_t block_end = std::min(k + block_size, n);

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

        for (std::size_t ii = block_end; ii < n; ii += block_size)
        {
            const std::size_t i_end = std::min(ii + block_size, n);

            for (std::size_t jj = ii; jj < n; jj += block_size)
            {
                const std::size_t j_end = std::min(jj + block_size, n);

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

    cholesky_detail::mirror_upper_to_lower(c, static_cast<int>(n));
}
