#ifndef CHOLESKY_VERSIONS_H
#define CHOLESKY_VERSIONS_H

#include "mphil_dis_cholesky.h"

#include <cstddef>

inline constexpr int kDefaultBlockedCholeskyBlockSize = 16;

// Each implementation factorises the matrix in place.
void cholesky_baseline(double *c, std::size_t n);
void cholesky_lower_triangle(double *c, std::size_t n);
void cholesky_upper_triangle(double *c, std::size_t n);
void cholesky_contiguous_access(double *c, std::size_t n);
void cholesky_cache_blocked_1(double *c, std::size_t n, std::size_t block_size);
void cholesky_cache_blocked_2(double *c, std::size_t n, std::size_t block_size);
void cholesky_openmp_1(double *c, std::size_t n);
void cholesky_openmp_2(double *c, std::size_t n);
void cholesky_openmp_3(double *c, std::size_t n, std::size_t block_size);
void cholesky_openmp_4(double *c, std::size_t n, std::size_t block_size);

#endif
