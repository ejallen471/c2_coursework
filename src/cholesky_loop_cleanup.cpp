#include "cholesky_helpers.h"
#include "cholesky_versions.h"

#include <cmath>

void cholesky_loop_cleanup(double* c, const std::size_t n)
{
    for (std::size_t p = 0; p < n; ++p)
    {
        const std::size_t p_row = p * n;
        const std::size_t p_add_one = p + 1;
        const std::size_t diag_index = p_row + p;

        const double current_diag = std::sqrt(c[diag_index]);
        c[diag_index] = current_diag;

        const double current_diag_recip = 1.0 / current_diag;

        for (std::size_t i = p_add_one; i < n; ++i)
        {
            const std::size_t i_row = i * n;
            c[i_row + p] *= current_diag_recip;
        }

        for (std::size_t j = p_add_one; j < n; ++j)
        {
            const std::size_t j_row = j * n;
            const double cjp = c[j_row + p];

            for (std::size_t i = j; i < n; ++i)
            {
                const std::size_t i_row = i * n;
                c[i_row + j] -= c[i_row + p] * cjp;
            }
        }
    }

    cholesky_detail::mirror_lower_to_upper(c, n);
}
