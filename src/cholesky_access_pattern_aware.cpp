#include "cholesky_helpers.h"
#include "cholesky_versions.h"

#include <cmath>

void cholesky_access_pattern_aware(double* c, const std::size_t n)
{
    for (std::size_t p = 0; p < n; ++p)
    {
        const std::size_t start = p + 1;
        double* row_p = c + p * n;

        const double diag = std::sqrt(row_p[p]);
        row_p[p] = diag;

        for (std::size_t i = start; i < n; ++i)
        {
            double* row_i = c + i * n;
            row_i[p] /= diag;
        }

        for (std::size_t i = start; i < n; ++i)
        {
            double* row_i = c + i * n;
            const double cip = row_i[p];

            for (std::size_t j = start; j <= i; ++j)
            {
                row_i[j] -= cip * c[j * n + p];
            }
        }
    }

    cholesky_detail::mirror_lower_to_upper(c, n);
}
