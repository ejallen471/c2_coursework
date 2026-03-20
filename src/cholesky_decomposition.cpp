/**
 * @file cholesky_decomposition.cpp
 * @brief Runtime dispatch and timing entry points for all Cholesky implementations.
 */

#include "cholesky_decomposition.h"
#include "cholesky_versions.h"
#include "timer.h"

#ifdef _OPENMP
#include <omp.h>
#endif

namespace
{
    constexpr double kInternalKernelFailureCode = -4.0;

    /**
     * @brief Measure the runtime of one factorisation callable.
     *
     * @param factorisation Callable that performs the factorisation.
     * @return Elapsed wall-clock time in seconds.
     */
    template <typename Factorisation>
    double time_factorisation(Factorisation factorisation)
    {
        const double t0 = wall_time_seconds();
        factorisation();
        const double t1 = wall_time_seconds();
        return t1 - t0;
    }

    /**
     * @brief Return the effective block size for a blocked implementation.
     *
     * @param version Implementation identifier.
     * @return Runtime-selected block size for that implementation.
     */
    bool version_uses_openmp(CholeskyVersion version)
    {
        switch (version)
        {
        case CholeskyVersion::OpenMPRowParallelUnblocked:
        case CholeskyVersion::OpenMPTileParallelBlocked:
        case CholeskyVersion::OpenMPBlockRowParallel:
        case CholeskyVersion::OpenMPTileListParallel:
        case CholeskyVersion::OpenMPTaskDAGBlocked:
            return true;

        default:
            return false;
        }
    }

    /**
     * @brief Return the effective block size for a blocked implementation.
     *
     * @param version Implementation identifier.
     * @param options Explicit runtime settings for the call.
     * @return Runtime-selected block size for that implementation.
     */
    std::size_t blocked_block_size_for(CholeskyVersion version,
                                       const CholeskyRuntimeOptions& options)
    {
        if (options.block_size > 0)
        {
            return static_cast<std::size_t>(options.block_size);
        }

        switch (version)
        {
        case CholeskyVersion::BlockedTileKernels:
        case CholeskyVersion::BlockedTileKernelsUnrolled:
        case CholeskyVersion::OpenMPTileParallelBlocked:
        case CholeskyVersion::OpenMPBlockRowParallel:
        case CholeskyVersion::OpenMPTileListParallel:
        case CholeskyVersion::OpenMPTaskDAGBlocked:
            return kDefaultBlockedCholeskyBlockSize;

        default:
            return kDefaultBlockedCholeskyBlockSize;
        }
    }
} // namespace

double timed_cholesky_factorisation_versioned_configured(double *c,
                                                         int n,
                                                         CholeskyVersion version,
                                                         const CholeskyRuntimeOptions& options)
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

#ifdef _OPENMP
    if (version_uses_openmp(version) && options.thread_count > 0)
    {
        omp_set_num_threads(options.thread_count);
    }
#endif

    switch (version)
    {
    case CholeskyVersion::Baseline:
        return time_factorisation([&]() { cholesky_baseline(c, matrix_size); });

    case CholeskyVersion::LowerTriangleOnly:
        return time_factorisation([&]() { cholesky_lower_triangle(c, matrix_size); });

    case CholeskyVersion::UpperTriangle:
        return time_factorisation([&]() { cholesky_upper_triangle(c, matrix_size); });

    case CholeskyVersion::ContiguousAccess:
        return time_factorisation([&]() { cholesky_contiguous_access(c, matrix_size); });

    case CholeskyVersion::BlockedTileKernels:
        return time_factorisation(
            [&]() {
                cholesky_blocked_tile_kernels(
                    c, matrix_size, blocked_block_size_for(version, options));
            });

    case CholeskyVersion::BlockedTileKernelsUnrolled:
        return time_factorisation(
            [&]() {
                cholesky_blocked_tile_kernels_unrolled(
                    c, matrix_size, blocked_block_size_for(version, options));
            });

    case CholeskyVersion::OpenMPRowParallelUnblocked:
        // Dispatch to the unblocked OpenMP kernel directly because its parallelism lives in
        // row-wise worksharing rather than in any caller-supplied block-size parameter.
        return time_factorisation([&]() { cholesky_openmp_row_parallel_unblocked(c, matrix_size); });

    case CholeskyVersion::OpenMPTileParallelBlocked:
        // Dispatch to the tile-parallel OpenMP kernel and resolve its tile size at runtime so
        // users can tune parallel work granularity without recompiling the library.
        return time_factorisation([&]() {
            cholesky_openmp_tile_parallel_blocked(
                c, matrix_size, blocked_block_size_for(version, options));
        });

    case CholeskyVersion::OpenMPBlockRowParallel:
        // Dispatch to the block-row OpenMP kernel with its resolved block size because row-level
        // load balance depends heavily on the chosen tile height.
        return time_factorisation([&]() {
            cholesky_openmp_block_row_parallel(
                c, matrix_size, blocked_block_size_for(version, options));
        });

    case CholeskyVersion::OpenMPTileListParallel:
        // Dispatch to the tile-list OpenMP kernel with a runtime-resolved tile size because
        // both cache reuse and work-list overhead are controlled by that setting.
        return time_factorisation([&]() {
            cholesky_openmp_tile_list_parallel(
                c, matrix_size, blocked_block_size_for(version, options));
        });

    case CholeskyVersion::OpenMPTaskDAGBlocked:
        // Dispatch to the task-DAG OpenMP kernel with a runtime-resolved block size because
        // task granularity directly controls both dependency overhead and available parallelism.
        {
            const double t0 = wall_time_seconds();
            const int status =
                cholesky_openmp_task_dag_blocked(c, matrix_size, blocked_block_size_for(version, options));
            const double t1 = wall_time_seconds();
            return (status == 0) ? (t1 - t0) : kInternalKernelFailureCode;
        }
    }

    return -3.0;
}

double timed_cholesky_factorisation_versioned(double *c, int n, CholeskyVersion version)
{
    const CholeskyRuntimeOptions options;
    return timed_cholesky_factorisation_versioned_configured(c, n, version, options);
}

double timed_cholesky_factorisation(double *c, int n)
{
    return timed_cholesky_factorisation_versioned(c, n, CholeskyVersion::Baseline);
}
