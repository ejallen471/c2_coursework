/**
 * @file 10_cholesky_openmp_task_dag_blocked.cpp
 * @brief Blocked OpenMP Cholesky using a simple task-based lower-triangular DAG.
 *
 * Strategy:
 * 1. Solve the diagonal block
 * 2. Solve the block column below it
 * 3. Update the trailing matrix with tasks
 *
 * This version is intentionally close to the faster lower-triangular task-based code.
 * The aim is to reduce overhead and keep the kernels simple.
 */

#include "cholesky_helpers.h"
#include "cholesky_versions.h"

#include <cmath>
#include <cstddef>
#include <cstdio>
#include <memory>

namespace
{
inline std::size_t min_sz(std::size_t a, std::size_t b)
{
    return (a < b) ? a : b;
}

inline std::size_t block_index(std::size_t r, std::size_t c, std::size_t nb)
{
    return r * nb + c;
}

/**
 * @brief Mirror the computed lower triangle into the upper triangle.
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
 * @brief Solve one diagonal block in lower-triangular storage.
 *
 * @return `0` on success, or `1` if a non-positive pivot is detected.
 */
int solve_diagonal_block(double* c, std::size_t n, std::size_t block_start, std::size_t block_end)
{
    for (std::size_t j = block_start; j < block_end; ++j)
    {
        double* row_j = c + j * n;
        double diag = row_j[j];

        double diag_sum = 0.0;
#pragma omp simd reduction(+ : diag_sum)
        for (std::size_t p = block_start; p < j; ++p)
        {
            diag_sum += row_j[p] * row_j[p];
        }

        diag -= diag_sum;

        if (diag <= 0.0)
        {
            std::fprintf(stderr,
                         "Error: openmp_task_dag_blocked non-positive pivot at index %zu: %.17g\n",
                         j,
                         diag);
            return 1;
        }

        diag = std::sqrt(diag);
        const double inv_diag = 1.0 / diag;
        row_j[j] = diag;

        for (std::size_t i = j + 1; i < block_end; ++i)
        {
            double* row_i = c + i * n;
            double s = row_i[j];

            double s_sum = 0.0;
#pragma omp simd reduction(+ : s_sum)
            for (std::size_t p = block_start; p < j; ++p)
            {
                s_sum += row_i[p] * row_j[p];
            }

            s -= s_sum;
            row_i[j] = s * inv_diag;
        }
    }

    return 0;
}

/**
 * @brief Solve one block column below the active diagonal block.
 */
void solve_block_column(double* c,
                        std::size_t n,
                        std::size_t block_start,
                        std::size_t block_end,
                        std::size_t row_block_start,
                        std::size_t row_block_end)
{
    for (std::size_t j = block_start; j < block_end; ++j)
    {
        double* row_j = c + j * n;
        const double inv_diag = 1.0 / row_j[j];

        for (std::size_t i = row_block_start; i < row_block_end; ++i)
        {
            double* row_i = c + i * n;
            double s = row_i[j];

            double s_sum = 0.0;
#pragma omp simd reduction(+ : s_sum)
            for (std::size_t p = block_start; p < j; ++p)
            {
                s_sum += row_i[p] * row_j[p];
            }

            s -= s_sum;
            row_i[j] = s * inv_diag;
        }
    }
}

/**
 * @brief Update one trailing diagonal block.
 */
void update_trailing_diagonal_block(double* c,
                                    std::size_t n,
                                    std::size_t block_start,
                                    std::size_t block_end,
                                    std::size_t row_block_start,
                                    std::size_t row_block_end)
{
    for (std::size_t i = row_block_start; i < row_block_end; ++i)
    {
        double* row_i = c + i * n;

        for (std::size_t j = row_block_start; j <= i; ++j)
        {
            double* row_j = c + j * n;
            double s = row_i[j];

            double s_sum = 0.0;
#pragma omp simd reduction(+ : s_sum)
            for (std::size_t p = block_start; p < block_end; ++p)
            {
                s_sum += row_i[p] * row_j[p];
            }

            s -= s_sum;
            row_i[j] = s;
        }
    }
}

/**
 * @brief Update one trailing off-diagonal square block.
 */
void update_trailing_square_block(double* c,
                                  std::size_t n,
                                  std::size_t block_start,
                                  std::size_t block_end,
                                  std::size_t row_block_start,
                                  std::size_t row_block_end,
                                  std::size_t col_block_start,
                                  std::size_t col_block_end)
{
    for (std::size_t i = row_block_start; i < row_block_end; ++i)
    {
        double* row_i = c + i * n;

        for (std::size_t j = col_block_start; j < col_block_end; ++j)
        {
            double* row_j = c + j * n;
            double s = row_i[j];

            double s_sum = 0.0;
#pragma omp simd reduction(+ : s_sum)
            for (std::size_t p = block_start; p < block_end; ++p)
            {
                s_sum += row_i[p] * row_j[p];
            }

            s -= s_sum;
            row_i[j] = s;
        }
    }
}
} // namespace

int cholesky_openmp_task_dag_blocked(double* c, std::size_t n, std::size_t block_size)
{
    if (c == nullptr || n == 0 || block_size == 0)
    {
        return 1;
    }

    const std::size_t num_blocks = (n + block_size - 1) / block_size;

    std::unique_ptr<int[]> deps_storage(new int[num_blocks * num_blocks]());
    int* dep_tokens = deps_storage.get();

    int errors = 0;

#pragma omp parallel
    {
#pragma omp single
        {
            for (std::size_t block = 0; block < num_blocks; ++block)
            {
                const std::size_t block_start = block * block_size;
                const std::size_t block_end = min_sz(block_start + block_size, n);

#pragma omp task firstprivate(block_start, block_end, block)                                       \
    depend(inout : dep_tokens[block_index(block, block, num_blocks)]) shared(c, errors)
                {
                    const int error = solve_diagonal_block(c, n, block_start, block_end);
#pragma omp atomic update
                    errors += error;
                }

                for (std::size_t row_block = block + 1; row_block < num_blocks; ++row_block)
                {
                    const std::size_t row_block_start = row_block * block_size;
                    const std::size_t row_block_end = min_sz(row_block_start + block_size, n);

#pragma omp task firstprivate(                                                                     \
        block_start, block_end, row_block_start, row_block_end, block, row_block)                  \
    depend(in : dep_tokens[block_index(block, block, num_blocks)])                                 \
    depend(inout : dep_tokens[block_index(row_block, block, num_blocks)]) shared(c, errors)
                    {
                        solve_block_column(
                            c, n, block_start, block_end, row_block_start, row_block_end);
                    }
                }

                for (std::size_t row_block = block + 1; row_block < num_blocks; ++row_block)
                {
                    const std::size_t row_block_start = row_block * block_size;
                    const std::size_t row_block_end = min_sz(row_block_start + block_size, n);

#pragma omp task firstprivate(                                                                     \
        block_start, block_end, row_block_start, row_block_end, block, row_block)                  \
    depend(in : dep_tokens[block_index(row_block, block, num_blocks)])                             \
    depend(inout : dep_tokens[block_index(row_block, row_block, num_blocks)]) shared(c, errors)
                    {
                        update_trailing_diagonal_block(
                            c, n, block_start, block_end, row_block_start, row_block_end);
                    }

                    for (std::size_t col_block = block + 1; col_block < row_block; ++col_block)
                    {
                        const std::size_t col_block_start = col_block * block_size;
                        const std::size_t col_block_end = min_sz(col_block_start + block_size, n);

#pragma omp task firstprivate(block_start,                                                         \
                                  block_end,                                                       \
                                  row_block_start,                                                 \
                                  row_block_end,                                                   \
                                  col_block_start,                                                 \
                                  col_block_end,                                                   \
                                  block,                                                           \
                                  row_block,                                                       \
                                  col_block)                                                       \
    depend(in : dep_tokens[block_index(row_block, block, num_blocks)],                             \
               dep_tokens[block_index(col_block, block, num_blocks)])                              \
    depend(inout : dep_tokens[block_index(row_block, col_block, num_blocks)]) shared(c, errors)
                        {
                            update_trailing_square_block(c,
                                                         n,
                                                         block_start,
                                                         block_end,
                                                         row_block_start,
                                                         row_block_end,
                                                         col_block_start,
                                                         col_block_end);
                        }
                    }
                }
            }

#pragma omp taskwait
        }
    }

    if (errors == 0)
    {
        mirror_lower_to_upper(c, n);
    }

    return errors;
}