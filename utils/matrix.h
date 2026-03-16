/**
 * @file matrix.h
 * @brief Helpers for constructing and validating dense SPD benchmark matrices.
 */

#ifndef MATRIX_H
#define MATRIX_H

#include <cstdint>
#include <vector>

/**
 * @struct MatrixGenerationOptions
 * @brief Configuration for the random SPD matrix generator.
 */
struct MatrixGenerationOptions
{
    std::uint64_t seed = 20260310ULL; ///< Seed used by the random number generator.
    double amplitude = 1.0;           ///< Magnitude bound for off-diagonal samples.
    double nugget = 1.0e-3;           ///< Positive diagonal slack added for strict dominance.
};

/**
 * @brief Checks whether a matrix is strictly diagonally dominant.
 * @param a Row-major matrix storage.
 * @param n Matrix dimension.
 * @return `true` when every diagonal entry exceeds the sum of off-diagonal magnitudes in its row.
 */
bool matrix_is_strictly_diagonally_dominant(const std::vector<double>& a, int n);

/**
 * @brief Verifies the structural conditions imposed by the generated SPD matrix builder.
 * @param a Row-major matrix storage.
 * @param n Matrix dimension.
 * @return `true` when the matrix is square, symmetric, positive on the diagonal, and diagonally dominant.
 */
bool matrix_satisfies_generated_spd_conditions(const std::vector<double>& a, int n);

/**
 * @brief Builds the coursework brief correlation matrix.
 * @param n Matrix dimension.
 * @return Dense row-major matrix storage.
 */
std::vector<double> make_coursework_brief_matrix(int n);

/**
 * @brief Returns a copy whose diagonal is increased to satisfy Gershgorin-style SPD checks.
 * @param a Source row-major matrix storage.
 * @param n Matrix dimension.
 * @return Adjusted matrix, or an empty vector when the input shape is invalid.
 */
std::vector<double> make_gershgorin_adjusted_copy(const std::vector<double>& a, int n);

/**
 * @brief Builds a reproducible dense SPD matrix with default generation options.
 * @param n Matrix dimension.
 * @return Dense row-major matrix storage.
 */
std::vector<double> make_generated_spd_matrix(int n);

/**
 * @brief Builds a reproducible dense SPD matrix with caller-supplied generation options.
 * @param n Matrix dimension.
 * @param options Generator configuration controlling the random seed and diagonal dominance margin.
 * @return Dense row-major matrix storage.
 */
std::vector<double> make_generated_spd_matrix(int n, const MatrixGenerationOptions& options);

#endif
