#include "cholesky_helpers.h"
#include "cholesky_versions.h"

#include <cmath>

void cholesky_lower_triangle_only(double* c, const std::size_t n)
{
    for (std::size_t p = 0; p < n; ++p)
    {
        const std::size_t p_row = p * n;
        const std::size_t next_diagonal_elem = p + 1;

        const double current_diag_elem = std::sqrt(c[p_row + p]);
        c[p_row + p] = current_diag_elem;

        const double current_diag_elem_recip = 1.0 / current_diag_elem;

        for (std::size_t i = next_diagonal_elem; i < n; ++i)
        {
            const std::size_t i_row = i * n;
            c[i_row + p] *= current_diag_elem_recip;
        }

        for (std::size_t j = next_diagonal_elem; j < n; ++j)
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
