#ifndef MPHIL_DIS_CHOLESKY_H
#define MPHIL_DIS_CHOLESKY_H

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

double mphil_dis_cholesky(double* c, int n);
double mphil_dis_cholesky_versioned(double* c, int n, CholeskyVersion version);

#endif
