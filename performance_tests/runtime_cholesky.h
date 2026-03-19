/**
 * @file runtime_cholesky.h
 * @brief Small helpers for mapping user input to actual Cholesky implementations.
 */

#ifndef RUNTIME_CHOLESKY_H
#define RUNTIME_CHOLESKY_H

#include "cholesky_decomposition.h"

#include <algorithm>
#include <cctype>
#include <string>

/**
 * @brief Clean up a user-provided optimisation name.
 *
 * This makes matching easier by:
 * - converting everything to lower case
 * - replacing spaces and hyphens with underscores
 *
 * @param name Raw input string.
 * @return Normalised name.
 */
inline std::string normalise_optimisation_name(std::string name)
{
    std::transform(name.begin(),
                   name.end(),
                   name.begin(),
                   [](unsigned char ch)
                   {
                       return static_cast<char>(std::tolower(ch));
                   });

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
 * @brief Convert a string into a Cholesky implementation enum.
 *
 * This is used when parsing command-line arguments.
 *
 * @param input User-provided name.
 * @param version Output enum if recognised.
 * @return `true` if the name is valid.
 */
inline bool parse_optimisation_name(const std::string& input, CholeskyVersion& version)
{
    const std::string name = normalise_optimisation_name(input);

    if (name == "baseline")
    {
        version = CholeskyVersion::Baseline;
        return true;
    }

    if (name == "lower_triangle" || name == "lowertriangle" || name == "lower_triangle_only" ||
        name == "lowertriangleonly" || name == "triangular_only")
    {
        version = CholeskyVersion::LowerTriangleOnly;
        return true;
    }

    if (name == "upper_triangle" || name == "uppertriangle" || name == "upper_triangle_only" ||
        name == "uppertriangleonly" || name == "triangular_upper" || name == "triangularupper")
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

    if (name == "cholesky_blocked_tile_kernels" || name == "choleskyblockedtilekernels" ||
        name == "blocked_tile_kernels" || name == "blockedtilekernels")
    {
        version = CholeskyVersion::BlockedTileKernels;
        return true;
    }

    if (name == "cholesky_blocked_tile_kernels_unrolled" ||
        name == "choleskyblockedtilekernelsunrolled" || name == "blocked_tile_kernels_unrolled" ||
        name == "blockedtilekernelsunrolled")
    {
        version = CholeskyVersion::BlockedTileKernelsUnrolled;
        return true;
    }

    if (name == "openmp_row_parallel_unblocked")
    {
        version = CholeskyVersion::OpenMPRowParallelUnblocked;
        return true;
    }

    if (name == "openmp_tile_parallel_blocked")
    {
        version = CholeskyVersion::OpenMPTileParallelBlocked;
        return true;
    }

    if (name == "openmp_block_row_parallel")
    {
        version = CholeskyVersion::OpenMPBlockRowParallel;
        return true;
    }

    if (name == "openmp_tile_list_parallel")
    {
        version = CholeskyVersion::OpenMPTileListParallel;
        return true;
    }

    if (name == "openmp_task_dag_blocked")
    {
        version = CholeskyVersion::OpenMPTaskDAGBlocked;
        return true;
    }

    return false;
}

/**
 * @brief Convert enum → string (used when writing CSVs).
 *
 * @param version Implementation identifier.
 * @return Canonical name.
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

        case CholeskyVersion::BlockedTileKernels:
            return "cholesky_blocked_tile_kernels";

        case CholeskyVersion::BlockedTileKernelsUnrolled:
            return "cholesky_blocked_tile_kernels_unrolled";

        case CholeskyVersion::OpenMPRowParallelUnblocked:
            return "openmp_row_parallel_unblocked";

        case CholeskyVersion::OpenMPTileParallelBlocked:
            return "openmp_tile_parallel_blocked";

        case CholeskyVersion::OpenMPBlockRowParallel:
            return "openmp_block_row_parallel";

        case CholeskyVersion::OpenMPTileListParallel:
            return "openmp_tile_list_parallel";

        case CholeskyVersion::OpenMPTaskDAGBlocked:
            return "openmp_task_dag_blocked";
    }

    return "unknown";
}

/**
 * @brief Check if a version uses OpenMP.
 */
bool optimisation_uses_openmp(CholeskyVersion version);

/**
 * @brief Check if a version supports block size tuning.
 */
bool optimisation_supports_block_size(CholeskyVersion version);

/**
 * @brief Run a chosen implementation (simple version).
 */
inline double run_cholesky_version(double* c, int n, CholeskyVersion version)
{
    return timed_cholesky_factorisation_versioned(c, n, version);
}

/**
 * @brief Run with explicit runtime options (threads + block size).
 *
 * This is used by the benchmarking code.
 */
inline double run_cholesky_version_configured(
    double* c, int n, CholeskyVersion version, int thread_count, int block_size)
{
    CholeskyRuntimeOptions options;
    options.thread_count = thread_count;
    options.block_size = block_size;

    return timed_cholesky_factorisation_versioned_configured(c, n, version, options);
}

#endif