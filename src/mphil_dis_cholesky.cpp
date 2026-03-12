/*
This file does the following

1. basic input validation checks
2. choose the configured implementation
3. run that implementation
4. return the runtime
*/

#include "mphil_dis_cholesky.h"
#include "cholesky_versions.h"
#include "timer.h"

namespace
{
// Map the compile-time CMake selection onto the internal implementation enum.
CholeskyVersion current_version()
{
#if !defined(MPHIL_CHOLESKY_VERSION_ID) || MPHIL_CHOLESKY_VERSION_ID == 0
    return CholeskyVersion::Baseline;
#elif MPHIL_CHOLESKY_VERSION_ID == 1
    return CholeskyVersion::LowerTriangleOnly;
#elif MPHIL_CHOLESKY_VERSION_ID == 2
    return CholeskyVersion::InlineMirror;
#elif MPHIL_CHOLESKY_VERSION_ID == 3
    return CholeskyVersion::LoopCleanup;
#elif MPHIL_CHOLESKY_VERSION_ID == 4
    return CholeskyVersion::AccessPatternAware;
#elif MPHIL_CHOLESKY_VERSION_ID == 5
    return CholeskyVersion::CacheBlocked;
#elif MPHIL_CHOLESKY_VERSION_ID == 6
    return CholeskyVersion::VectorFriendly;
#elif MPHIL_CHOLESKY_VERSION_ID == 7
    return CholeskyVersion::BlockedVectorised;
#elif MPHIL_CHOLESKY_VERSION_ID == 8
    return CholeskyVersion::OpenMP1;
#elif MPHIL_CHOLESKY_VERSION_ID == 9
    return CholeskyVersion::OpenMP2;
#elif MPHIL_CHOLESKY_VERSION_ID == 10
    return CholeskyVersion::OpenMP3;
#else
#error "Invalid MPHIL_CHOLESKY_VERSION_ID"
#endif
}

} // namespace

double mphil_dis_cholesky(double* c, int n)
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

    // Time only the factorisation itself.
    const double t0 = wall_time_seconds();

    // Dispatch to the selected implementation. All versions factorise in place.
    switch (current_version())
    {
    case CholeskyVersion::Baseline:
        cholesky_baseline(c, n);
        break;

    case CholeskyVersion::LowerTriangleOnly:
        cholesky_lower_triangle_only(c, n);
        break;

    case CholeskyVersion::InlineMirror:
        cholesky_inline_mirror(c, n);
        break;

    case CholeskyVersion::LoopCleanup:
        cholesky_loop_cleanup(c, n);
        break;

    case CholeskyVersion::AccessPatternAware:
        cholesky_access_pattern_aware(c, n);
        break;

    case CholeskyVersion::CacheBlocked:
    {
        cholesky_cache_blocked(c, n, kDefaultBlockedCholeskyBlockSize);
        break;
    }

    case CholeskyVersion::VectorFriendly:
        cholesky_vectorisation(c, n);
        break;

    case CholeskyVersion::BlockedVectorised:
    {
        cholesky_blocked_vectorised(c, n, kDefaultBlockedCholeskyBlockSize);
        break;
    }

    case CholeskyVersion::OpenMP1:
#if defined(MPHIL_HAVE_OPENMP) && MPHIL_HAVE_OPENMP
        cholesky_openmp_1(c, n);
        break;
#else
        return -4.0;
#endif

    case CholeskyVersion::OpenMP2:
        return -4.0;

    case CholeskyVersion::OpenMP3:
        return -4.0;

    default:
        // This should not happen if the compile-time version mapping is valid.
        return -3.0;
    }

    const double t1 = wall_time_seconds();

    // Return elapsed wall-clock time in seconds.
    return t1 - t0;
}
