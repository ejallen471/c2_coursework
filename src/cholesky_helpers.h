/**
 * @file cholesky_helpers.h
 * @brief Internal helpers for restoring mirrored matrix storage after factorisation.
 */

#ifndef CHOLESKY_HELPERS_H
#define CHOLESKY_HELPERS_H

#include <cstddef>

/**
 * @namespace cholesky_detail
 * @brief Internal routines shared by Cholesky implementations.
 */
namespace cholesky_detail
{
    /**
     * @brief Copies the computed lower triangle into the upper triangle.
     * @param c Pointer to row-major matrix storage.
     * @param n Matrix dimension.
     */
    inline void mirror_lower_to_upper(double *c, std::size_t n)
    {
        for (std::size_t i = 0; i < n; ++i)
        {
            double *dst = c + i * n + i + 1;

            for (std::size_t j = i + 1; j < n; ++j)
            {
                const double *row_j = c + j * n;
                *dst++ = row_j[i];
            }
        }
    }

    /**
     * @brief Copies the computed upper triangle into the lower triangle.
     * @param c Pointer to row-major matrix storage.
     * @param n Matrix dimension.
     */
    inline void mirror_upper_to_lower(double *c, std::size_t n)
    {

        for (std::size_t i = 1; i < n; ++i)
        {
            double *row_i = c + i * n;

            for (std::size_t j = 0; j < i; ++j)
            {
                row_i[j] = c[j * n + i];
            }
        }
    }
} // namespace cholesky_detail

#endif
