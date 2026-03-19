/**
 * @file cholesky_decomposition.h
 * @brief Public interface for the timed Cholesky factorisation library.
 */

#ifndef COURSEWORK_CHOLESKY_LIBRARY_H
#define COURSEWORK_CHOLESKY_LIBRARY_H

/**
 * @defgroup LibraryAPI Library API
 * @brief Functions for running and timing Cholesky factorisations.
 *
 * All functions take a Symmetric Positive Definite (SPD) matrix stored in row-major format.
 * The matrix is overwritten in place with its Cholesky factor.
 * Each function returns the elapsed runtime in seconds.
 *
 * The versioned interface allows the user to choose between different implementations,
 * including single-threaded and mutli-threaded (openMP) variants.
 * @{
 */

/**
 * @enum CholeskyVersion
 * @brief Available Cholesky implementations.
 */
enum class CholeskyVersion
{
    Baseline,                   ///< Simple reference implementation updating both triangles.
    LowerTriangleOnly,          ///< Computes lower triangle and mirrors to full matrix.
    UpperTriangle,              ///< Computes upper triangle and mirrors to full matrix.
    ContiguousAccess,           ///< Single-threaded version with contiguous memory access.
    BlockedTileKernels,         ///< Blocked version using clean tile-based structure.
    BlockedTileKernelsUnrolled, ///< Blocked version with manual loop unrolling.
    OpenMPRowParallelUnblocked, ///< OpenMP version parallelising rows (no blocking).
    OpenMPTileParallelBlocked,  ///< OpenMP blocked version parallelising tiles.
    OpenMPBlockRowParallel,     ///< OpenMP blocked version parallelising block rows.
    OpenMPTileListParallel,     ///< OpenMP version using an explicit list of tile work.
    OpenMPTaskDAGBlocked        ///< OpenMP version using a task-based DAG.
};

/**
 * @brief Optional runtime settings for one timed factorisation call.
 *
 * A zero value keeps the built-in default for that setting.
 */
struct CholeskyRuntimeOptions
{
    int block_size = 0;   ///< Explicit block size for blocked kernels, or `0` for the built-in default.
    int thread_count = 0; ///< Explicit OpenMP thread count, or `0` to keep the current runtime setting.
};

/**
 * @brief Run and time the baseline Cholesky factorisation.
 *
 * @param c Pointer to an `n x n` row-major SPD matrix (modified in place).
 * @param n Matrix size.
 * @return Elapsed time in seconds (or a negative value if an error occurs).
 *
 * This is the simplest entry point and always uses the baseline implementation.
 * Use this when you want a stable reference for comparison.
 */
double timed_cholesky_factorisation(double* c, int n);

/**
 * @brief Run and time a chosen Cholesky implementation.
 *
 * @param c Pointer to an `n x n` row-major SPD matrix (modified in place).
 * @param n Matrix size.
 * @param version Which implementation to use.
 * @return Elapsed time in seconds, or a negative value if an error occurs.
 *
 * The selected implementation can be any of the following:
 *
 * - `baseline`
 * - `lower_triangle`
 * - `upper_triangle`
 * - `contiguous_access`
 * - `cholesky_blocked_tile_kernels`
 * - `cholesky_blocked_tile_kernels_unrolled`
 * - `openmp_row_parallel_unblocked`
 * - `openmp_tile_parallel_blocked`
 * - `openmp_block_row_parallel`
 * - `openmp_tile_list_parallel`
 * - `openmp_task_dag_blocked`
 *
 */
double timed_cholesky_factorisation_versioned(double* c, int n, CholeskyVersion version);

/**
 * @brief Run and time a chosen Cholesky implementation with explicit runtime settings.
 *
 * @param c Pointer to an `n x n` row-major SPD matrix (modified in place).
 * @param n Matrix size.
 * @param version Which implementation to use.
 * @param options Explicit block-size and thread-count settings for this call.
 * @return Elapsed time in seconds, or a negative value if an error occurs.
 *
 * Blocked methods use `options.block_size` when it is positive, otherwise the built-in default.
 * OpenMP methods use `options.thread_count` when it is positive, otherwise the current runtime setting.
 */
double timed_cholesky_factorisation_versioned_configured(double* c,
                                                         int n,
                                                         CholeskyVersion version,
                                                         const CholeskyRuntimeOptions& options);

/** @} */

#endif
