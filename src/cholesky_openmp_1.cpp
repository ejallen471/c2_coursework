#include "cholesky_helpers.h"
#include "cholesky_versions.h"

#include <cmath>
#include <cstddef>

void cholesky_openmp_1(double* c, int n)
{
    const std::size_t n_size = static_cast<std::size_t>(n);

    for (int p = 0; p < n; ++p)
    {
        const std::size_t p_index = static_cast<std::size_t>(p);
        const std::size_t p_row = p_index * n_size;
        const int start = p + 1;

        const double diag = std::sqrt(c[p_row + p_index]);
        c[p_row + p_index] = diag;

#pragma omp parallel for schedule(static)
        for (int i = start; i < n; ++i)
        {
            const std::size_t i_row = static_cast<std::size_t>(i) * n_size;
            c[i_row + p_index] /= diag;
        }

#pragma omp parallel for schedule(static)
        for (int i = start; i < n; ++i)
        {
            const std::size_t i_row = static_cast<std::size_t>(i) * n_size;
            const double cip = c[i_row + p_index];

            for (int j = start; j <= i; ++j)
            {
                const std::size_t j_row = static_cast<std::size_t>(j) * n_size;
                c[i_row + static_cast<std::size_t>(j)] -= cip * c[j_row + p_index];
            }
        }
    }

    cholesky_detail::mirror_lower_to_upper(c, n_size);
}
