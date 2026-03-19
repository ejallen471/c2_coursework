/**
 * @file cholesky_versions.h
 * @brief Declarations for all Cholesky factorisation implementations.
 */

#ifndef CHOLESKY_VERSIONS_H
#define CHOLESKY_VERSIONS_H

#include "cholesky_decomposition.h"

#include <cstddef>

/**
 * @defgroup FactorisationKernels Factorisation Kernels
 * @brief Internal kernel entry points used by the runtime dispatcher and benchmark drivers.
 *
 * These declarations expose the concrete implementations behind the public versioned API.
 * They are useful for benchmark drivers and tests, but most users should prefer
 * `timed_cholesky_factorisation_versioned()` from the public library header.
 * @{
 */

/// Default tile size used by blocked implementations when no override is supplied.
inline constexpr int kDefaultBlockedCholeskyBlockSize = 16;

/**
 * @brief Baseline in-place Cholesky factorisation.
 * @ingroup FactorisationKernels
 * @param c Pointer to row-major matrix storage.
 * @param n Matrix dimension.
 */
void cholesky_baseline(double *c, std::size_t n);

/**
 * @brief Lower-triangular in-place Cholesky factorisation mirrored into full storage.
 * @ingroup FactorisationKernels
 * @param c Pointer to row-major matrix storage.
 * @param n Matrix dimension.
 */
void cholesky_lower_triangle(double *c, std::size_t n);

/**
 * @brief Upper-triangular in-place Cholesky factorisation mirrored into full storage.
 * @ingroup FactorisationKernels
 * @param c Pointer to row-major matrix storage.
 * @param n Matrix dimension.
 */
void cholesky_upper_triangle(double *c, std::size_t n);

/**
 * @brief Single-threaded factorisation tuned for contiguous row-major access.
 * @ingroup FactorisationKernels
 * @param c Pointer to row-major matrix storage.
 * @param n Matrix dimension.
 */
void cholesky_contiguous_access(double *c, std::size_t n);

/**
 * @brief First blocked single-threaded factorisation variant.
 * @ingroup FactorisationKernels
 * @param c Pointer to row-major matrix storage.
 * @param n Matrix dimension.
 * @param block_size Tile size used by the blocked update kernels.
 */
void cholesky_blocked_tile_kernels(double *c, std::size_t n, std::size_t block_size);

/**
 * @brief Second blocked single-threaded factorisation variant with loop unrolling.
 * @ingroup FactorisationKernels
 * @param c Pointer to row-major matrix storage.
 * @param n Matrix dimension.
 * @param block_size Tile size used by the blocked update kernels.
 */
void cholesky_blocked_tile_kernels_unrolled(double *c, std::size_t n, std::size_t block_size);

/**
 * @brief OpenMP factorisation that parallelises trailing row updates.
 * @ingroup FactorisationKernels
 * @param c Pointer to row-major matrix storage.
 * @param n Matrix dimension.
 *
 * @details This unblocked OpenMP path keeps pivot construction serial and shares the trailing
 * row-update work across the active OpenMP team.
 */
void cholesky_openmp_row_parallel_unblocked(double *c, std::size_t n);

/**
 * @brief OpenMP blocked factorisation with tile-parallel trailing updates.
 * @ingroup FactorisationKernels
 * @param c Pointer to row-major matrix storage.
 * @param n Matrix dimension.
 * @param block_size Tile size used by the blocked update kernels.
 *
 * @details This variant factors one panel at a time and parallelises the trailing matrix update
 * at tile granularity, which usually makes it a good default blocked OpenMP implementation.
 */
void cholesky_openmp_tile_parallel_blocked(double *c, std::size_t n, std::size_t block_size);

/**
 * @brief OpenMP blocked factorisation with block-row-parallel trailing updates.
 * @ingroup FactorisationKernels
 * @param c Pointer to row-major matrix storage.
 * @param n Matrix dimension.
 * @param block_size Tile size used by the blocked update kernels.
 *
 * @details This variant distributes whole trailing block rows across threads, favouring simpler
 * scheduling over the more fine-grained tile decomposition used by other OpenMP variants.
 */
void cholesky_openmp_block_row_parallel(double *c, std::size_t n, std::size_t block_size);

/**
 * @brief OpenMP blocked factorisation with explicit tile work lists.
 * @ingroup FactorisationKernels
 * @param c Pointer to row-major matrix storage.
 * @param n Matrix dimension.
 * @param block_size Tile size used by the blocked update kernels.
 *
 * @details This variant constructs explicit panel and trailing tile lists before dispatching the
 * heavier trailing updates in parallel, making the scheduling policy more predictable.
 */
void cholesky_openmp_tile_list_parallel(double *c, std::size_t n, std::size_t block_size);

/**
 * @brief OpenMP blocked factorisation with a tiled POTRF/TRSM/SYRK/GEMM task DAG.
 * @ingroup FactorisationKernels
 * @param c Pointer to row-major matrix storage.
 * @param n Matrix dimension.
 * @param block_size Tile size used by the blocked update kernels.
 *
 * @details This task-based variant keeps the panel work serial and represents the trailing update
 * as a dependency-managed task graph so different tiles can proceed as soon as their inputs exist.
 */
void cholesky_openmp_task_dag_blocked(double *c, std::size_t n, std::size_t block_size);

/** @} */

#endif
