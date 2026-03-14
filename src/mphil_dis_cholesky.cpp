/*
This file does the following

1. basic input validation checks
2. choose the requested implementation
3. run that implementation
4. return the runtime
*/

#include "mphil_dis_cholesky.h"
#include "cholesky_versions.h"
#include "timer.h"

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
        cholesky_cache_blocked_1(c, matrix_size, kDefaultBlockedCholeskyBlockSize);
        const double t1 = wall_time_seconds();
        return t1 - t0;
    }

    case CholeskyVersion::cacheBlockedTwo:
    {
        const double t0 = wall_time_seconds();
        cholesky_cache_blocked_2(c, matrix_size, kDefaultBlockedCholeskyBlockSize);
        const double t1 = wall_time_seconds();
        return t1 - t0;
    }

    case CholeskyVersion::OpenMP1:
    {
#if defined(MPHIL_HAVE_OPENMP) && MPHIL_HAVE_OPENMP
        const double t0 = wall_time_seconds();
        cholesky_openmp_1(c, matrix_size);
        const double t1 = wall_time_seconds();
        return t1 - t0;
#else
        return -4.0;
#endif
    }

    case CholeskyVersion::OpenMP2:
    {
#if defined(MPHIL_HAVE_OPENMP) && MPHIL_HAVE_OPENMP
        const double t0 = wall_time_seconds();
        cholesky_openmp_2(c, matrix_size);
        const double t1 = wall_time_seconds();
        return t1 - t0;
#else
        return -4.0;
#endif
    }

    case CholeskyVersion::OpenMP3:
    {
#if defined(MPHIL_HAVE_OPENMP) && MPHIL_HAVE_OPENMP
        const double t0 = wall_time_seconds();
        cholesky_openmp_3(c, matrix_size, kDefaultBlockedCholeskyBlockSize);
        const double t1 = wall_time_seconds();
        return t1 - t0;
#else
        return -4.0;
#endif
    }

    case CholeskyVersion::OpenMP4:
    {
#if defined(MPHIL_HAVE_OPENMP) && MPHIL_HAVE_OPENMP
        const double t0 = wall_time_seconds();
        cholesky_openmp_4(c, matrix_size, kDefaultBlockedCholeskyBlockSize);
        const double t1 = wall_time_seconds();
        return t1 - t0;
#else
        return -4.0;
#endif
    }
    }

    return -3.0;
}

double timed_cholesky_factorisation(double *c, int n)
{
    return timed_cholesky_factorisation_versioned(c, n, CholeskyVersion::Baseline);
}
