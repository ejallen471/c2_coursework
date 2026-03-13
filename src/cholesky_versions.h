#ifndef CHOLESKY_VERSIONS_H
#define CHOLESKY_VERSIONS_H

#include "mphil_dis_cholesky.h"

#include <cstddef>

inline constexpr int kDefaultBlockedCholeskyBlockSize = 16;

// Each implementation factorises the matrix in place. Timing callers should derive any benchmark
// guard from the output matrix after the timed region.
void cholesky_baseline(double* c, std::size_t n);
void cholesky_lower_triangle_only(double* c, std::size_t n);
void cholesky_inline_mirror(double* c, std::size_t n);
void cholesky_loop_cleanup(double* c, std::size_t n);
void cholesky_access_pattern_aware(double* c, std::size_t n);
void cholesky_cache_blocked(double* c, std::size_t n, int block_size);
void cholesky_vectorisation(double* c, std::size_t n);
void cholesky_blocked_vectorised(double* c, std::size_t n, int block_size);
void cholesky_openmp_1(double* c, int n);
void cholesky_openmp_2(double* c, int n);
void cholesky_openmp_3(double* c, int n);

#endif
