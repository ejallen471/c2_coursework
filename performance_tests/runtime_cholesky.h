#ifndef RUNTIME_CHOLESKY_H
#define RUNTIME_CHOLESKY_H

#include "mphil_dis_cholesky.h"

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

    if (name == "lower_triangle" || name == "lowertriangle" ||
        name == "lower_triangle_only" || name == "lowertriangleonly" ||
        name == "triangular_only")
    {
        version = CholeskyVersion::LowerTriangleOnly;
        return true;
    }

    if (name == "upper_triangle" || name == "uppertriangle" ||
        name == "upper_triangle_only" || name == "uppertriangleonly" ||
        name == "triangular_upper" || name == "triangularupper")
    {
        version = CholeskyVersion::UpperTriangle;
        return true;
    }

    if (name == "contiguous_access" || name == "contiguousaccess" ||
        name == "access_pattern_aware" || name == "accesspatternaware" ||
        name == "access_pattern" || name == "accesspattern")
    {
        version = CholeskyVersion::ContiguousAccess;
        return true;
    }

    if (name == "cache_blocked_1" || name == "cacheblocked_1" ||
        name == "cache_blocked1" || name == "cacheblocked1")
    {
        version = CholeskyVersion::CacheBlocked;
        return true;
    }

    if (name == "cache_blocked_2" || name == "cacheblocked_2" ||
        name == "cache_blocked2" || name == "cacheblocked2")
    {
        version = CholeskyVersion::BlockedOptimal;
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
        return "lower_triangle";

    case CholeskyVersion::UpperTriangle:
        return "upper_triangle";

    case CholeskyVersion::ContiguousAccess:
        return "contiguous_access";

    case CholeskyVersion::CacheBlocked:
        return "cache_blocked_1";

    case CholeskyVersion::BlockedOptimal:
        return "cache_blocked_2";

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
    return timed_cholesky_factorisation_versioned(c, n, version);
}

#endif
