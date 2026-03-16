/**
 * @file mphil_dis_cholesky.cpp
 * @brief Runtime dispatch and timing entry points for all Cholesky implementations.
 */

/*
This file does the following

1. Basic input validation checks
2. choose the requested implementation
3. run that implementation
4. return the runtime
*/

#include "mphil_dis_cholesky.h"
#include "cholesky_versions.h"
#include "timer.h"

#include <cstdlib>

namespace
{
std::size_t parse_positive_block_size_env(const char* name)
{
    const char* value = std::getenv(name);
    if (value == nullptr || *value == '\0')
    {
        return 0;
    }

    char* end = nullptr;
    const long parsed = std::strtol(value, &end, 10);
    if (end == value || *end != '\0' || parsed <= 0)
    {
        return 0;
    }

    return static_cast<std::size_t>(parsed);
}

std::size_t blocked_block_size_for(CholeskyVersion version)
{
    const std::size_t generic_override = parse_positive_block_size_env("MPHIL_BLOCK_SIZE");

    switch (version)
    {
    case CholeskyVersion::cacheBlockedOne:
    {
        const std::size_t specific =
            parse_positive_block_size_env("MPHIL_CACHE_BLOCKED_1_BLOCK_SIZE");
        return (specific > 0) ? specific
                              : ((generic_override > 0) ? generic_override
                                                        : kDefaultBlockedCholeskyBlockSize);
    }

    case CholeskyVersion::cacheBlockedTwo:
    {
        const std::size_t specific =
            parse_positive_block_size_env("MPHIL_CACHE_BLOCKED_2_BLOCK_SIZE");
        return (specific > 0) ? specific
                              : ((generic_override > 0) ? generic_override
                                                        : kDefaultBlockedCholeskyBlockSize);
    }

    case CholeskyVersion::OpenMP3:
    {
        const std::size_t specific = parse_positive_block_size_env("MPHIL_OPENMP3_BLOCK_SIZE");
        return (specific > 0) ? specific
                              : ((generic_override > 0) ? generic_override
                                                        : kDefaultBlockedCholeskyBlockSize);
    }

    case CholeskyVersion::OpenMP4:
    {
        const std::size_t specific = parse_positive_block_size_env("MPHIL_OPENMP4_BLOCK_SIZE");
        return (specific > 0) ? specific
                              : ((generic_override > 0) ? generic_override
                                                        : kDefaultBlockedCholeskyBlockSize);
    }

    default:
        return kDefaultBlockedCholeskyBlockSize;
    }
}
} // namespace

double timed_cholesky_factorisation_versioned(double *c, int n, CholeskyVersion version)
{
    // Reject invalid pointers before touching matrix storage.
    if (c == nullptr)
    {
        return -1.0;
    }

    // Keep aligned with the size bound.
    if (n <= 0 || n > 100000)
    {
        return -2.0;
    }

    const std::size_t matrix_size = static_cast<std::size_t>(n);

    switch (version)
    {
    case CholeskyVersion::Baseline:
    {
        const double t0 = wall_time_seconds();
        cholesky_baseline(c, matrix_size);
        const double t1 = wall_time_seconds();
        return t1 - t0;
    }

    case CholeskyVersion::LowerTriangleOnly:
    {
        const double t0 = wall_time_seconds();
        cholesky_lower_triangle(c, matrix_size);
        const double t1 = wall_time_seconds();
        return t1 - t0;
    }

    case CholeskyVersion::UpperTriangle:
    {
        const double t0 = wall_time_seconds();
        cholesky_upper_triangle(c, matrix_size);
        const double t1 = wall_time_seconds();
        return t1 - t0;
    }

    case CholeskyVersion::ContiguousAccess:
    {
        const double t0 = wall_time_seconds();
        cholesky_contiguous_access(c, matrix_size);
        const double t1 = wall_time_seconds();
        return t1 - t0;
    }

    case CholeskyVersion::cacheBlockedOne:
    {
        const double t0 = wall_time_seconds();
        cholesky_cache_blocked_1(c, matrix_size, blocked_block_size_for(version));
        const double t1 = wall_time_seconds();
        return t1 - t0;
    }

    case CholeskyVersion::cacheBlockedTwo:
    {
        const double t0 = wall_time_seconds();
        cholesky_cache_blocked_2(c, matrix_size, blocked_block_size_for(version));
        const double t1 = wall_time_seconds();
        return t1 - t0;
    }

    case CholeskyVersion::OpenMP1:
    {
        const double t0 = wall_time_seconds();
        cholesky_openmp_1(c, matrix_size);
        const double t1 = wall_time_seconds();
        return t1 - t0;
    }

    case CholeskyVersion::OpenMP2:
    {
        const double t0 = wall_time_seconds();
        cholesky_openmp_2(c, matrix_size);
        const double t1 = wall_time_seconds();
        return t1 - t0;
    }

    case CholeskyVersion::OpenMP3:
    {
        const double t0 = wall_time_seconds();
        cholesky_openmp_3(c, matrix_size, blocked_block_size_for(version));
        const double t1 = wall_time_seconds();
        return t1 - t0;
    }

    case CholeskyVersion::OpenMP4:
    {
        const double t0 = wall_time_seconds();
        cholesky_openmp_4(c, matrix_size, blocked_block_size_for(version));
        const double t1 = wall_time_seconds();
        return t1 - t0;
    }
    }

    return -3.0;
}

double timed_cholesky_factorisation(double *c, int n)
{
    return timed_cholesky_factorisation_versioned(c, n, CholeskyVersion::Baseline);
}
