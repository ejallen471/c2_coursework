#include "cholesky_helpers.h"
#include "cholesky_versions.h"

#include <algorithm>
#include <cmath>
#include <vector>

void cholesky_blocked_vectorised(double* c, const std::size_t n, const int block_size)
{
    std::vector<double> diag_recip(static_cast<std::size_t>(block_size), 0.0);

    for (std::size_t k = 0; k < n; k += static_cast<std::size_t>(block_size))
    {
        const std::size_t block_end =
            std::min(k + static_cast<std::size_t>(block_size), n);

        for (std::size_t p = k; p < block_end; ++p)
        {
            double* row_p = c + p * n;
            const std::size_t start = p + 1;

            const double diag = std::sqrt(row_p[p]);
            row_p[p] = diag;

            const double inv_diag = 1.0 / diag;
            diag_recip[p - k] = inv_diag;

            for (std::size_t i = start; i < block_end; ++i)
            {
                double* row_i = c + i * n;
                row_i[p] *= inv_diag;
            }

            for (std::size_t i = start; i < block_end; ++i)
            {
                double* row_i = c + i * n;
                const double cip = row_i[p];

                std::size_t j = start;
                for (; j + 3 <= i; j += 4)
                {
                    row_i[j] -= cip * c[j * n + p];
                    row_i[j + 1] -= cip * c[(j + 1) * n + p];
                    row_i[j + 2] -= cip * c[(j + 2) * n + p];
                    row_i[j + 3] -= cip * c[(j + 3) * n + p];
                }

                for (; j <= i; ++j)
                {
                    row_i[j] -= cip * c[j * n + p];
                }
            }
        }

        for (std::size_t i = block_end; i < n; ++i)
        {
            double* row_i = c + i * n;

            for (std::size_t p = k; p < block_end; ++p)
            {
                const double* row_p = c + p * n;
                double sum = row_i[p];

                std::size_t s = k;
                for (; s + 3 < p; s += 4)
                {
                    sum -= row_i[s] * row_p[s];
                    sum -= row_i[s + 1] * row_p[s + 1];
                    sum -= row_i[s + 2] * row_p[s + 2];
                    sum -= row_i[s + 3] * row_p[s + 3];
                }

                for (; s < p; ++s)
                {
                    sum -= row_i[s] * row_p[s];
                }

                row_i[p] = sum * diag_recip[p - k];
            }
        }

        for (std::size_t ii = block_end; ii < n; ii += static_cast<std::size_t>(block_size))
        {
            const std::size_t i_end =
                std::min(ii + static_cast<std::size_t>(block_size), n);

            for (std::size_t jj = block_end; jj < ii; jj += static_cast<std::size_t>(block_size))
            {
                const std::size_t j_end =
                    std::min(jj + static_cast<std::size_t>(block_size), n);

                for (std::size_t i = ii; i < i_end; ++i)
                {
                    double* row_i = c + i * n;

                    std::size_t j = jj;
                    for (; j + 3 < j_end; j += 4)
                    {
                        const double* row_j0 = c + j * n;
                        const double* row_j1 = c + (j + 1) * n;
                        const double* row_j2 = c + (j + 2) * n;
                        const double* row_j3 = c + (j + 3) * n;

                        double sum0 = 0.0;
                        double sum1 = 0.0;
                        double sum2 = 0.0;
                        double sum3 = 0.0;

                        for (std::size_t p = k; p < block_end; ++p)
                        {
                            const double aip = row_i[p];
                            sum0 += aip * row_j0[p];
                            sum1 += aip * row_j1[p];
                            sum2 += aip * row_j2[p];
                            sum3 += aip * row_j3[p];
                        }

                        row_i[j] -= sum0;
                        row_i[j + 1] -= sum1;
                        row_i[j + 2] -= sum2;
                        row_i[j + 3] -= sum3;
                    }

                    for (; j < j_end; ++j)
                    {
                        const double* row_j = c + j * n;
                        double sum = 0.0;

                        for (std::size_t p = k; p < block_end; ++p)
                        {
                            sum += row_i[p] * row_j[p];
                        }

                        row_i[j] -= sum;
                    }
                }
            }

            for (std::size_t i = ii; i < i_end; ++i)
            {
                double* row_i = c + i * n;

                std::size_t j = ii;
                for (; j + 3 <= i; j += 4)
                {
                    const double* row_j0 = c + j * n;
                    const double* row_j1 = c + (j + 1) * n;
                    const double* row_j2 = c + (j + 2) * n;
                    const double* row_j3 = c + (j + 3) * n;

                    double sum0 = 0.0;
                    double sum1 = 0.0;
                    double sum2 = 0.0;
                    double sum3 = 0.0;

                    for (std::size_t p = k; p < block_end; ++p)
                    {
                        const double aip = row_i[p];
                        sum0 += aip * row_j0[p];
                        sum1 += aip * row_j1[p];
                        sum2 += aip * row_j2[p];
                        sum3 += aip * row_j3[p];
                    }

                    row_i[j] -= sum0;
                    row_i[j + 1] -= sum1;
                    row_i[j + 2] -= sum2;
                    row_i[j + 3] -= sum3;
                }

                for (; j <= i; ++j)
                {
                    const double* row_j = c + j * n;
                    double sum = 0.0;

                    for (std::size_t p = k; p < block_end; ++p)
                    {
                        sum += row_i[p] * row_j[p];
                    }

                    row_i[j] -= sum;
                }
            }
        }
    }

    cholesky_detail::mirror_lower_to_upper(c, n);
}
