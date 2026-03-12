#ifndef CHOLESKY_HELPERS_H
#define CHOLESKY_HELPERS_H

#include <cstddef>

namespace cholesky_detail
{
inline void mirror_lower_to_upper(double* c, int n)
{
    const std::size_t n_size = static_cast<std::size_t>(n);

    for (int i = 0; i < n; ++i)
    {
        double* dst = c + static_cast<std::size_t>(i) * n_size + static_cast<std::size_t>(i + 1);

        for (int j = i + 1; j < n; ++j)
        {
            const double* row_j = c + static_cast<std::size_t>(j) * n_size;
            *dst++ = row_j[i];
        }
    }
}
} // namespace cholesky_detail

#endif
