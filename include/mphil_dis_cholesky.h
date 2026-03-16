/**
 * @file mphil_dis_cholesky.h
 * @brief Public entry points for the timed Cholesky factorisation library.
 */

#ifndef MPHIL_DIS_CHOLESKY_H
#define MPHIL_DIS_CHOLESKY_H

/**
 * @enum CholeskyVersion
 * @brief Runtime-selectable Cholesky factorisation implementations.
 */
enum class CholeskyVersion
{
    Baseline,          ///< Baseline in-place factorisation that updates both triangles.
    LowerTriangleOnly, ///< Lower-triangular variant mirrored back into full storage.
    UpperTriangle,     ///< Upper-triangular variant aligned with row-major storage.
    ContiguousAccess,  ///< Cache-friendlier single-threaded variant with contiguous updates.
    cacheBlockedOne,   ///< First cache-blocked single-threaded implementation.
    cacheBlockedTwo,   ///< Second cache-blocked single-threaded implementation.
    OpenMP1,           ///< OpenMP variant with parallel trailing-row updates.
    OpenMP2,           ///< OpenMP variant with parallel row scaling and updates.
    OpenMP3,           ///< OpenMP blocked implementation with dynamic tile updates.
    OpenMP4            ///< OpenMP blocked implementation with explicit tile work lists.
};

/**
 * @brief Times the baseline Cholesky factorisation.
 * @param c Pointer to an `n x n` row-major SPD matrix overwritten in place.
 * @param n Matrix dimension.
 * @return Elapsed time in seconds, or a negative error code when validation fails.
 */
double timed_cholesky_factorisation(double* c, int n);

/**
 * @brief Times a selected Cholesky factorisation variant.
 * @param c Pointer to an `n x n` row-major SPD matrix overwritten in place.
 * @param n Matrix dimension.
 * @param version Runtime-selected implementation to execute.
 * @return Elapsed time in seconds, or a negative error code when validation fails.
 */
double timed_cholesky_factorisation_versioned(double* c, int n, CholeskyVersion version);

#endif
