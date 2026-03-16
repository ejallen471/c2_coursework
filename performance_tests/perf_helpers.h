/**
 * @file perf_helpers.h
 * @brief Shared helper utilities for benchmark reporting and validation.
 */

#ifndef PERF_HELPERS_H
#define PERF_HELPERS_H

#include <cstddef>
#include <filesystem>
#include <limits>
#include <string>
#include <vector>

/// Numeric type used for log-determinant calculations and CSV output.
using LogDetValue = long double;

/// Precision used when serialising `LogDetValue` results.
inline constexpr int kLogDetOutputPrecision = std::numeric_limits<LogDetValue>::max_digits10;

/**
 * @brief Computes `log(det(A))` from a Cholesky factor stored in mirrored matrix storage.
 * @param c Factorised matrix storage.
 * @param n Matrix dimension.
 * @return Log-determinant reconstructed from the diagonal of the factor.
 */
LogDetValue logdet_from_factorised_storage(const std::vector<double>& c, std::size_t n);

/**
 * @brief Computes the relative percentage difference between two values.
 * @param value Measured value.
 * @param reference Reference value.
 * @return Percentage difference relative to the magnitude of `reference`.
 */
LogDetValue relative_difference_percent(LogDetValue value, LogDetValue reference);

/**
 * @brief Reconstructs the original matrix from mirrored Cholesky storage.
 * @param c Factorised matrix storage.
 * @param n Matrix dimension.
 * @return Reconstructed dense row-major matrix.
 */
std::vector<double> reconstruct_from_factorised_storage(const std::vector<double>& c, std::size_t n);

/**
 * @brief Computes the relative Frobenius norm error between two matrices.
 * @param actual Measured matrix values.
 * @param reference Reference matrix values.
 * @return Relative Frobenius error, or infinity for mismatched shapes.
 */
double relative_frobenius_error(const std::vector<double>& actual,
                                const std::vector<double>& reference);

/**
 * @brief Computes a reference log-determinant with LAPACK when available.
 * @param c Copy of the source matrix, passed by value because LAPACK factorises in place.
 * @param n Matrix dimension.
 * @param logdet Output parameter populated on success.
 * @return `true` when LAPACK support is available and the factorisation succeeds.
 */
bool lapack_reference_logdet(std::vector<double> c, int n, LogDetValue& logdet);

/**
 * @brief Wraps a filesystem path in double quotes for shell-safe display.
 * @param path Path to quote.
 * @return Quoted string form of `path`.
 */
std::string quoted_path(const std::filesystem::path& path);

#endif
