#ifndef RUNTIME_CHOLESKY_H
#define RUNTIME_CHOLESKY_H

#include "cholesky_versions.h"
#include "timer.h"

#include <algorithm>
#include <cctype>
#include <string>

inline std::string normalise_optimisation_name(std::string name)
{
    std::transform(name.begin(),
                   name.end(),
                   name.begin(),
                   [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });

    for (char& ch : name)
    {
        if (ch == '-' || ch == ' ')
        {
            ch = '_';
        }
    }

    return name;
}

inline bool parse_optimisation_name(const std::string& input, CholeskyVersion& version)
{
    const std::string name = normalise_optimisation_name(input);

    if (name == "baseline")
    {
        version = CholeskyVersion::Baseline;
        return true;
    }

    if (name == "lower_triangle_only" || name == "lowertriangleonly" ||
        name == "triangular_only")
    {
        version = CholeskyVersion::LowerTriangleOnly;
        return true;
    }

    if (name == "inline_mirror" || name == "inlinemirror" || name == "mirror_in_loop" ||
        name == "mirror_within_update")
    {
        version = CholeskyVersion::InlineMirror;
        return true;
    }

    if (name == "loop_cleanup" || name == "loopcleanup")
    {
        version = CholeskyVersion::LoopCleanup;
        return true;
    }

    if (name == "access_pattern" || name == "access_pattern_aware" ||
        name == "accesspatternaware")
    {
        version = CholeskyVersion::AccessPatternAware;
        return true;
    }

    if (name == "cache_blocked" || name == "cacheblocked")
    {
        version = CholeskyVersion::CacheBlocked;
        return true;
    }

    if (name == "vectorisation" || name == "vector_friendly" || name == "vectorfriendly")
    {
        version = CholeskyVersion::VectorFriendly;
        return true;
    }

    if (name == "blocked_vectorised" || name == "blocked_vectorized" ||
        name == "blockedvectorised" || name == "blockedvectorized" ||
        name == "cache_blocked_vectorised" || name == "cache_blocked_vectorized")
    {
        version = CholeskyVersion::BlockedVectorised;
        return true;
    }

    if (name == "openmp1")
    {
        version = CholeskyVersion::OpenMP1;
        return true;
    }

    if (name == "openmp2")
    {
        version = CholeskyVersion::OpenMP2;
        return true;
    }

    if (name == "openmp3")
    {
        version = CholeskyVersion::OpenMP3;
        return true;
    }

    return false;
}

inline const char* optimisation_name(CholeskyVersion version)
{
    switch (version)
    {
    case CholeskyVersion::Baseline:
        return "baseline";

    case CholeskyVersion::LowerTriangleOnly:
        return "lower_triangle_only";

    case CholeskyVersion::InlineMirror:
        return "inline_mirror";

    case CholeskyVersion::LoopCleanup:
        return "loop_cleanup";

    case CholeskyVersion::AccessPatternAware:
        return "access_pattern";

    case CholeskyVersion::CacheBlocked:
        return "cache_blocked";

    case CholeskyVersion::VectorFriendly:
        return "vectorisation";

    case CholeskyVersion::BlockedVectorised:
        return "blocked_vectorised";

    case CholeskyVersion::OpenMP1:
        return "openmp1";

    case CholeskyVersion::OpenMP2:
        return "openmp2";

    case CholeskyVersion::OpenMP3:
        return "openmp3";
    }

    return "unknown";
}

inline double run_cholesky_version(double* c, int n, CholeskyVersion version)
{
    if (c == nullptr)
    {
        return -1.0;
    }

    if (n <= 0 || n > 100000)
    {
        return -2.0;
    }

    const double t0 = wall_time_seconds();

    switch (version)
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
        cholesky_cache_blocked(c, n, kDefaultBlockedCholeskyBlockSize);
        break;

    case CholeskyVersion::VectorFriendly:
        cholesky_vectorisation(c, n);
        break;

    case CholeskyVersion::BlockedVectorised:
        cholesky_blocked_vectorised(c, n, kDefaultBlockedCholeskyBlockSize);
        break;

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
    }

    const double t1 = wall_time_seconds();
    return t1 - t0;
}

#endif
