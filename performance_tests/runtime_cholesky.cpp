/**
 * @file runtime_cholesky.cpp
 * @brief Small helper functions used to choose how each Cholesky version is run.
 */

#include "runtime_cholesky.h"

/**
 * @brief Check whether an implementation uses OpenMP.
 *
 * This is used to decide whether thread-related settings should be applied.
 *
 * @param version The implementation to check.
 * @return `true` if the implementation uses OpenMP.
 */
bool optimisation_uses_openmp(CholeskyVersion version)
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
 * @brief Check whether an implementation supports a tunable block size.
 *
 * Some implementations are blocked and allow a block size to be set,
 * while others ignore this parameter.
 *
 * @param version The implementation to check.
 * @return `true` if the implementation uses a block size.
 */
bool optimisation_supports_block_size(CholeskyVersion version)
{
    switch (version)
    {
        case CholeskyVersion::BlockedTileKernels:
        case CholeskyVersion::BlockedTileKernelsUnrolled:
        case CholeskyVersion::OpenMPTileParallelBlocked:
        case CholeskyVersion::OpenMPBlockRowParallel:
        case CholeskyVersion::OpenMPTileListParallel:
        case CholeskyVersion::OpenMPTaskDAGBlocked:
            return true;

        default:
            return false;
    }
}