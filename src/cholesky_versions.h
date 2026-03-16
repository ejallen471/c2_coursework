/**
 * @file cholesky_versions.h
 * @brief Declarations for all Cholesky factorisation implementations.
 */

#ifndef CHOLESKY_VERSIONS_H
#define CHOLESKY_VERSIONS_H

#include "mphil_dis_cholesky.h"

#include <cstddef>

/// Default tile size used by blocked implementations when no override is supplied.
inline constexpr int kDefaultBlockedCholeskyBlockSize = 16;

/**
 * @brief Baseline in-place Cholesky factorisation.
 * @param c Pointer to row-major matrix storage.
 * @param n Matrix dimension.
 */
void cholesky_baseline(double *c, std::size_t n);

/**
 * @brief Lower-triangular in-place Cholesky factorisation mirrored into full storage.
 * @param c Pointer to row-major matrix storage.
 * @param n Matrix dimension.
 */
void cholesky_lower_triangle(double *c, std::size_t n);

/**
 * @brief Upper-triangular in-place Cholesky factorisation mirrored into full storage.
 * @param c Pointer to row-major matrix storage.
 * @param n Matrix dimension.
 */
void cholesky_upper_triangle(double *c, std::size_t n);

/**
 * @brief Single-threaded factorisation tuned for contiguous row-major access.
 * @param c Pointer to row-major matrix storage.
 * @param n Matrix dimension.
 */
void cholesky_contiguous_access(double *c, std::size_t n);

/**
 * @brief First blocked single-threaded factorisation variant.
 * @param c Pointer to row-major matrix storage.
 * @param n Matrix dimension.
 * @param block_size Tile size used by the blocked update kernels.
 */
void cholesky_cache_blocked_1(double *c, std::size_t n, std::size_t block_size);

/**
 * @brief Second blocked single-threaded factorisation variant with loop unrolling.
 * @param c Pointer to row-major matrix storage.
 * @param n Matrix dimension.
 * @param block_size Tile size used by the blocked update kernels.
 */
void cholesky_cache_blocked_2(double *c, std::size_t n, std::size_t block_size);

/**
 * @brief OpenMP factorisation that parallelises trailing row updates.
 * @param c Pointer to row-major matrix storage.
 * @param n Matrix dimension.
 */
void cholesky_openmp_1(double *c, std::size_t n);

/**
 * @brief OpenMP factorisation that parallelises row scaling and trailing updates.
 * @param c Pointer to row-major matrix storage.
 * @param n Matrix dimension.
 */
void cholesky_openmp_2(double *c, std::size_t n);

/**
 * @brief OpenMP blocked factorisation with dynamic scheduling over trailing tiles.
 * @param c Pointer to row-major matrix storage.
 * @param n Matrix dimension.
 * @param block_size Tile size used by the blocked update kernels.
 */
void cholesky_openmp_3(double *c, std::size_t n, std::size_t block_size);

/**
 * @brief OpenMP blocked factorisation with explicit tile work lists.
 * @param c Pointer to row-major matrix storage.
 * @param n Matrix dimension.
 * @param block_size Tile size used by the blocked update kernels.
 */
void cholesky_openmp_4(double *c, std::size_t n, std::size_t block_size);

#endif
