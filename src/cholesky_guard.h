#ifndef CHOLESKY_GUARD_H
#define CHOLESKY_GUARD_H

#include <cstddef>

namespace cholesky_detail
{
inline volatile double cholesky_guard_sink = 0.0;

inline double factorised_matrix_guard(const double* c, std::size_t n)
{
    if (c == nullptr || n == 0)
    {
        return 0.0;
    }

    const std::size_t mid = n / 2;
    const std::size_t last = n - 1;
    double guard = c[0];
    guard += 3.0 * c[mid * n + mid];
    guard += 7.0 * c[last * n + last];

    if (mid > 0)
    {
        guard += c[mid * n + (mid - 1)];
    }

    if (last > mid)
    {
        guard += c[last * n + mid];
    }

    return guard;
}

inline void consume_cholesky_guard(double guard)
{
    cholesky_guard_sink = guard;
}
} // namespace cholesky_detail

#endif
