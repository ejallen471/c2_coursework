#include "cholesky_helpers.h"
#include "cholesky_versions.h"

#include <cmath>

void cholesky_cache_blocked(double* c, const std::size_t n, const int block_size)
{
    for (std::size_t k = 0; k < n; k += static_cast<std::size_t>(block_size))
    {
        const int b =
            ((k + static_cast<std::size_t>(block_size)) < n) ? block_size : static_cast<int>(n - k);
        const int block_end_idx = static_cast<int>(k) + b;

        for (std::size_t p = k; p < static_cast<std::size_t>(block_end_idx); ++p)
        {
            const std::size_t p_row = p * n;
            const int start = static_cast<int>(p) + 1;

            const double diag = std::sqrt(c[p_row + p]);
            c[p_row + p] = diag;

            for (int i = start; i < block_end_idx; ++i)
            {
                const std::size_t i_row = static_cast<std::size_t>(i) * n;
                c[i_row + p] /= diag;
            }

            for (std::size_t j = static_cast<std::size_t>(start);
                 j < static_cast<std::size_t>(block_end_idx);
                 ++j)
            {
                const std::size_t j_row = j * n;
                const double cjp = c[j_row + p];

                for (std::size_t i = j; i < static_cast<std::size_t>(block_end_idx); ++i)
                {
                    const std::size_t i_row = i * n;
                    c[i_row + j] -= c[i_row + p] * cjp;
                }
            }
        }

        for (std::size_t i = static_cast<std::size_t>(block_end_idx); i < n; ++i)
        {
            const std::size_t i_row = i * n;

            for (std::size_t p = k; p < static_cast<std::size_t>(block_end_idx); ++p)
            {
                const std::size_t p_row = p * n;
                double sum = c[i_row + p];

                for (std::size_t s = k; s < p; ++s)
                {
                    sum -= c[i_row + s] * c[p_row + s];
                }

                sum /= c[p_row + p];
                c[i_row + p] = sum;
            }
        }

        for (int ii = block_end_idx; ii < static_cast<int>(n); ii += block_size)
        {
            const int i_end = ((ii + block_size) < static_cast<int>(n)) ? (ii + block_size) :
                                                                       static_cast<int>(n);

            for (int jj = block_end_idx; jj <= ii; jj += block_size)
            {
                const int j_end = ((jj + block_size) < static_cast<int>(n)) ? (jj + block_size) :
                                                                           static_cast<int>(n);

                for (int i = ii; i < i_end; ++i)
                {
                    const std::size_t i_row = static_cast<std::size_t>(i) * n;
                    const int j_limit = (jj == ii) ? (i + 1) : j_end;

                    for (int j = jj; j < j_limit; ++j)
                    {
                        const std::size_t j_index = static_cast<std::size_t>(j);
                        const std::size_t j_row = j_index * n;
                        double sum = 0.0;

                        for (int p = static_cast<int>(k); p < block_end_idx; ++p)
                        {
                            const std::size_t p_index = static_cast<std::size_t>(p);
                            sum += c[i_row + p_index] * c[j_row + p_index];
                        }

                        c[i_row + j_index] -= sum;
                    }
                }
            }
        }
    }

    cholesky_detail::mirror_lower_to_upper(c, n);
}
