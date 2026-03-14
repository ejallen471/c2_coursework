#ifndef MPHIL_DIS_CHOLESKY_H
#define MPHIL_DIS_CHOLESKY_H

enum class CholeskyVersion
{
    Baseline,
    LowerTriangleOnly,
    UpperTriangle,
    ContiguousAccess,
    cacheBlockedOne,
    cacheBlockedTwo,
    OpenMP1,
    OpenMP2,
    OpenMP3,
    OpenMP4
};

double timed_cholesky_factorisation(double* c, int n);
double timed_cholesky_factorisation_versioned(double* c, int n, CholeskyVersion version);

#endif
