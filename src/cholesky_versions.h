#ifndef CHOLESKY_VERSIONS_H
#define CHOLESKY_VERSIONS_H

#include <cstddef>

enum class CholeskyVersion
{
    Baseline,
    LowerTriangleOnly,
    InlineMirror,
    LoopCleanup,
    AccessPatternAware,
    CacheBlocked,
    VectorFriendly,
    BlockedVectorised,
    OpenMP1,
    OpenMP2,
    OpenMP3
};

inline constexpr int kDefaultBlockedCholeskyBlockSize = 16;

void cholesky_baseline(double *c, std::size_t n);

#endif
