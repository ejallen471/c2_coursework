/**
 * @file runtime_cholesky.h
 * @brief Helpers for mapping user-facing optimisation names to runtime implementations.
 */

#ifndef RUNTIME_CHOLESKY_H
#define RUNTIME_CHOLESKY_H

#include "mphil_dis_cholesky.h"

#include <algorithm>
#include <cctype>
#include <string>

/**
 * @brief Normalises a user-supplied optimisation name for lookup.
 * @param name Raw optimisation name copied by value.
 * @return Lower-cased name with spaces and hyphens converted to underscores.
 */
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

/**
 * @brief Parses a user-facing optimisation label into a `CholeskyVersion`.
 * @param input Raw optimisation name.
 * @param version Output parameter populated when parsing succeeds.
 * @return `true` when the input maps to a known implementation.
 */
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
        version = CholeskyVersion::cacheBlockedOne;
        return true;
    }

    if (name == "cache_blocked_2" || name == "cacheblocked_2" ||
        name == "cache_blocked2" || name == "cacheblocked2")
    {
        version = CholeskyVersion::cacheBlockedTwo;
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

    if (name == "openmp4")
    {
        version = CholeskyVersion::OpenMP4;
        return true;
    }

    return false;
}

/**
 * @brief Returns the canonical command-line name for a Cholesky implementation.
 * @param version Implementation identifier.
 * @return Canonical lower-case name understood by the benchmark drivers.
 */
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

    case CholeskyVersion::cacheBlockedOne:
        return "cache_blocked_1";

    case CholeskyVersion::cacheBlockedTwo:
        return "cache_blocked_2";

    case CholeskyVersion::OpenMP1:
        return "openmp1";

    case CholeskyVersion::OpenMP2:
        return "openmp2";

    case CholeskyVersion::OpenMP3:
        return "openmp3";

    case CholeskyVersion::OpenMP4:
        return "openmp4";
    }

    return "unknown";
}

/**
 * @brief Runs the selected implementation through the timed library entry point.
 * @param c Pointer to row-major matrix storage.
 * @param n Matrix dimension.
 * @param version Implementation to execute.
 * @return Elapsed time in seconds, or a negative error code on failure.
 */
inline double run_cholesky_version(double* c, int n, CholeskyVersion version)
{
    return timed_cholesky_factorisation_versioned(c, n, version);
}

#endif
