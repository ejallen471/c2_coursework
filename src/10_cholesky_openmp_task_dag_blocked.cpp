/**
 * @file 10_cholesky_openmp_task_dag_blocked.cpp
 * @brief Blocked OpenMP Cholesky using tasks only for trailing updates.
 *
 * Strategy:
 * 1. Factor diagonal block (serial)
 * 2. Solve block column below (serial)
 * 3. Update trailing matrix using parallel tasks (O(n^3) work only)
 */

#include "cholesky_helpers.h"
#include "cholesky_versions.h"

#include <cmath>
#include <cstddef>
#include <memory>

namespace
{
/**
 * @brief Return the smaller of two tile bounds.
 *
 * @param a First bound.
 * @param b Second bound.
 * @return Smaller bound.
 */
inline std::size_t min_sz(std::size_t a, std::size_t b)
{
    return (a < b) ? a : b;
}

/**
 * @brief Flatten a tile coordinate into a dependency-token index.
 *
 * @param r Tile row index.
 * @param c Tile column index.
 * @param nb Number of block rows or columns.
 * @return Flat dependency-token index.
 */
inline std::size_t block_index(std::size_t r, std::size_t c, std::size_t nb)
{
    return r * nb + c;
}

/**
 * @brief Mirror the lower triangle into the upper triangle.
 *
 * @param c Row-major matrix storage updated in place.
 * @param n Matrix dimension.
 */
void mirror_lower_to_upper(double* c, std::size_t n)
{
    for (std::size_t i = 0; i < n; ++i)
    {
        double* row_i = c + i * n;
        for (std::size_t j = 0; j < i; ++j)
        {
            double* row_j = c + j * n;
            row_j[i] = row_i[j];
        }
    }
}

/**
 * @brief Factor one diagonal block in lower-triangular storage.
 *
 * @param c Row-major matrix storage updated in place.
 * @param n Matrix dimension.
 * @param bs Block start index.
 * @param be Block end index.
 * @return `0` on success, or `1` if a non-positive pivot is detected.
 *
 * @note The input matrix is assumed to be symmetric positive definite.
 */
int solve_diagonal_block(double* c, std::size_t n, std::size_t bs, std::size_t be)
{
    for (std::size_t j = bs; j < be; ++j)
    {
        double* row_j = c + j * n;
        double diag = row_j[j];

        // Ask OpenMP to vectorise the pivot correction because each subtraction is
        // independent and the reduction safely combines the SIMD lanes into `diag`.
#pragma omp simd reduction(- : diag)
        for (std::size_t p = bs; p < j; ++p)
        {
            diag -= row_j[p] * row_j[p];
        }

        if (diag <= 0.0)
            return 1;

        diag = std::sqrt(diag);
        const double inv = 1.0 / diag;
        row_j[j] = diag;

        for (std::size_t i = j + 1; i < be; ++i)
        {
            double* row_i = c + i * n;
            double s = row_i[j];

            // Ask OpenMP to vectorise the block-local forward solve because the
            // multiply-subtract terms are independent contributions to `s`.
#pragma omp simd reduction(- : s)
            for (std::size_t p = bs; p < j; ++p)
            {
                s -= row_i[p] * row_j[p];
            }

            row_i[j] = s * inv;
        }
    }
    return 0;
}

/**
 * @brief Solve one block column below the active diagonal block.
 *
 * @param c Row-major matrix storage updated in place.
 * @param n Matrix dimension.
 * @param bs Block start index.
 * @param be Block end index.
 * @param rs Row-block start index.
 * @param re Row-block end index.
 */
void solve_block_column(
    double* c, std::size_t n, std::size_t bs, std::size_t be, std::size_t rs, std::size_t re)
{
    for (std::size_t j = bs; j < be; ++j)
    {
        double* row_j = c + j * n;
        const double inv = 1.0 / row_j[j];

        for (std::size_t i = rs; i < re; ++i)
        {
            double* row_i = c + i * n;
            double s = row_i[j];

            // Ask OpenMP to vectorise the panel solve for the same reason: the reduction
            // captures the independent updates over the already computed panel columns.
#pragma omp simd reduction(- : s)
            for (std::size_t p = bs; p < j; ++p)
            {
                s -= row_i[p] * row_j[p];
            }

            row_i[j] = s * inv;
        }
    }
}

/**
 * @brief Update one trailing diagonal tile in lower-triangular storage.
 *
 * @param c Row-major matrix storage updated in place.
 * @param n Matrix dimension.
 * @param bs Active block start index.
 * @param be Active block end index.
 * @param rs Tile start row.
 * @param re Tile end row.
 */
void update_diag(
    double* c, std::size_t n, std::size_t bs, std::size_t be, std::size_t rs, std::size_t re)
{
    for (std::size_t i = rs; i < re; ++i)
    {
        double* row_i = c + i * n;

        for (std::size_t j = rs; j <= i; ++j)
        {
            double* row_j = c + j * n;
            double s = row_i[j];

            // Ask OpenMP to vectorise the diagonal-tile update because each panel-column
            // contribution is independent and can be folded into `s` via reduction.
#pragma omp simd reduction(- : s)
            for (std::size_t p = bs; p < be; ++p)
            {
                s -= row_i[p] * row_j[p];
            }

            row_i[j] = s;
        }
    }
}

/**
 * @brief Update one trailing off-diagonal tile in lower-triangular storage.
 *
 * @param c Row-major matrix storage updated in place.
 * @param n Matrix dimension.
 * @param bs Active block start index.
 * @param be Active block end index.
 * @param rs Tile start row.
 * @param re Tile end row.
 * @param cs Tile start column.
 * @param ce Tile end column.
 */
void update_offdiag(double* c,
                    std::size_t n,
                    std::size_t bs,
                    std::size_t be,
                    std::size_t rs,
                    std::size_t re,
                    std::size_t cs,
                    std::size_t ce)
{
    for (std::size_t i = rs; i < re; ++i)
    {
        double* row_i = c + i * n;

        for (std::size_t j = cs; j < ce; ++j)
        {
            double* row_j = c + j * n;
            double s = row_i[j];

            // Ask OpenMP to vectorise the off-diagonal tile update because the panel-width
            // loop is again a pure reduction with no cross-iteration write dependency.
#pragma omp simd reduction(- : s)
            for (std::size_t p = bs; p < be; ++p)
            {
                s -= row_i[p] * row_j[p];
            }

            row_i[j] = s;
        }
    }
}
} // namespace

void cholesky_openmp_task_dag_blocked(double* c, std::size_t n, std::size_t block_size)
{
    if (!c || n == 0 || block_size == 0)
        return;

    const std::size_t nb = (n + block_size - 1) / block_size;

    // Allocate one dependency token per tile so OpenMP task dependencies can name stable
    // memory locations and therefore serialise repeated updates to the same trailing tile.
    std::unique_ptr<int[]> deps(new int[nb * nb]());
    int* dep = deps.get();

    // Share a single error flag across the task-producing thread so we stop issuing work as
    // soon as a non-SPD pivot is detected instead of letting later tasks consume bad state.
    int error_flag = 0;

    // Create the OpenMP team once so one producer thread can emit tasks while the rest of the
    // workers execute the expensive trailing updates in parallel.
#pragma omp parallel
    {
        // Restrict task creation to one thread because the diagonal and panel phases must
        // remain serial, and duplicate producers would create the same tasks twice.
#pragma omp single
        {
            for (std::size_t b = 0; b < nb; ++b)
            {
                const std::size_t bs = b * block_size;
                const std::size_t be = min_sz(bs + block_size, n);

                // 1. Diagonal (serial)
                if (error_flag == 0)
                {
                    if (solve_diagonal_block(c, n, bs, be))
                        error_flag = 1;
                }

                if (error_flag)
                    break;

                // 2. Panel (serial)
                for (std::size_t rb = b + 1; rb < nb; ++rb)
                {
                    const std::size_t rs = rb * block_size;
                    const std::size_t re = min_sz(rs + block_size, n);

                    solve_block_column(c, n, bs, be, rs, re);
                }

                // 3. Trailing updates (tasks only)
                for (std::size_t rb = b + 1; rb < nb; ++rb)
                {
                    const std::size_t rs = rb * block_size;
                    const std::size_t re = min_sz(rs + block_size, n);

                    // Emit one task per diagonal trailing tile and depend on that tile's token
                    // so updates from successive panels are applied in the correct order.
#pragma omp task firstprivate(bs, be, rs, re, rb) depend(inout : dep[block_index(rb, rb, nb)])
                    {
                        update_diag(c, n, bs, be, rs, re);
                    }

                    for (std::size_t cb = b + 1; cb < rb; ++cb)
                    {
                        const std::size_t cs = cb * block_size;
                        const std::size_t ce = min_sz(cs + block_size, n);

                        // Emit one task per off-diagonal trailing tile and reuse the tile token
                        // to prevent multiple tasks from racing on the same output tile.
#pragma omp task firstprivate(bs, be, rs, re, cs, ce, rb, cb)                                      \
    depend(inout : dep[block_index(rb, cb, nb)])
                        {
                            update_offdiag(c, n, bs, be, rs, re, cs, ce);
                        }
                    }
                }

                // Wait for all updates generated from this panel before moving to the next
                // diagonal block, because the next panel depends on those updated tiles.
#pragma omp taskwait
            }
        }
    }

    // Mirror the completed lower triangle only after the OpenMP region has finished so the
    // symmetric copy cannot race with any outstanding update task.
    if (error_flag == 0)
        mirror_lower_to_upper(c, n);
}
