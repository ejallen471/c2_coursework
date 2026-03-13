#include "cholesky_versions.h"

#include <cmath>

void cholesky_inline_mirror(double* c, std::size_t n)
{
    for (std::size_t p = 0; p < n; ++p)
    {
        const std::size_t p_row = p * n;
        const std::size_t start = p + 1;

        const double diag = std::sqrt(c[p_row + p]);
        c[p_row + p] = diag;

        const double diag_recip = 1.0 / diag;

        for (std::size_t i = start; i < n; ++i)
        {
            const std::size_t i_row = i * n;
            c[i_row + p] *= diag_recip;
            c[p_row + i] = c[i_row + p];
        }

        for (std::size_t j = start; j < n; ++j)
        {
            const std::size_t j_row = j * n;
            const double cjp = c[j_row + p];

            for (std::size_t k = j; k < n; ++k)
            {
                const std::size_t k_row = k * n;
                c[k_row + j] -= c[k_row + p] * cjp;
                c[j_row + k] = c[k_row + j];
            }
        }
    }
}
