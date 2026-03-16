/**
 * @file test_helpers.h
 * @brief Shared assertions and matrix utilities used by the unit tests.
 */

#ifndef TEST_HELPERS_H
#define TEST_HELPERS_H

#include "matrix.h"

#include <vector>

/**
 * @brief Compares two floating-point values with combined absolute and relative tolerances.
 * @param a First value.
 * @param b Second value.
 * @param atol Absolute tolerance.
 * @param rtol Relative tolerance.
 * @return `true` when the values are sufficiently close.
 */
bool nearly_equal(double a, double b, double atol = 1e-12, double rtol = 1e-12);

/**
 * @brief Compares two vectors element-wise using `nearly_equal`.
 * @param a First vector.
 * @param b Second vector.
 * @param atol Absolute tolerance.
 * @param rtol Relative tolerance.
 * @return `true` when the vectors have matching size and close entries.
 */
bool vectors_close(const std::vector<double>& a, const std::vector<double>& b, double atol = 1e-12,
                   double rtol = 1e-12);

/**
 * @brief Checks whether a matrix is symmetric within tolerance.
 * @param a Row-major matrix storage.
 * @param n Matrix dimension.
 * @param atol Absolute tolerance.
 * @param rtol Relative tolerance.
 * @return `true` when the matrix is symmetric within the supplied tolerances.
 */
bool matrix_is_symmetric(const std::vector<double>& a, int n, double atol = 1e-12,
                         double rtol = 1e-12);

/**
 * @brief Checks whether all diagonal entries are strictly positive.
 * @param a Row-major matrix storage.
 * @param n Matrix dimension.
 * @return `true` when every diagonal entry is positive.
 */
bool diagonal_is_positive(const std::vector<double>& a, int n);

/**
 * @brief Prints a matrix to standard output in row-major order.
 * @param a Row-major matrix storage.
 * @param n Matrix dimension.
 */
void print_matrix(const std::vector<double>& a, int n);

/**
 * @brief Builds an identity matrix.
 * @param n Matrix dimension.
 * @return Dense row-major identity matrix.
 */
std::vector<double> make_identity_matrix(int n);

/**
 * @brief Builds a diagonal matrix from the supplied diagonal entries.
 * @param diag Diagonal values.
 * @return Dense row-major diagonal matrix.
 */
std::vector<double> make_diagonal_matrix(const std::vector<double>& diag);

/**
 * @brief Returns the small worked example matrix used in several correctness tests.
 * @return Dense `2 x 2` SPD matrix storage.
 */
std::vector<double> make_brief_example_matrix();

/**
 * @brief Extracts the lower-triangular factor from mirrored Cholesky storage.
 * @param c Factorised matrix storage.
 * @param n Matrix dimension.
 * @return Dense lower-triangular matrix.
 */
std::vector<double> lower_factor_from_storage(const std::vector<double>& c, int n);

/**
 * @brief Reconstructs the original matrix from mirrored Cholesky storage.
 * @param c Factorised matrix storage.
 * @param n Matrix dimension.
 * @return Reconstructed dense row-major matrix.
 */
std::vector<double> reconstruct_from_factorised_storage(const std::vector<double>& c, int n);

/**
 * @brief Computes the log-determinant from mirrored Cholesky storage.
 * @param c Factorised matrix storage.
 * @param n Matrix dimension.
 * @return Log-determinant of the original matrix.
 */
double logdet_from_factorised_storage(const std::vector<double>& c, int n);

#endif
