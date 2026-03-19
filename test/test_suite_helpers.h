/**
 * @file test_suite_helpers.h
 * @brief Shared helpers for the broader correctness and integration-oriented C++ tests.
 */

#ifndef TEST_SUITE_HELPERS_H
#define TEST_SUITE_HELPERS_H

#include "cholesky_versions.h"
#include "matrix.h"
#include "perf_helpers.h"
#include "runtime_cholesky.h"

#include <filesystem>
#include <map>
#include <string>
#include <utility>
#include <vector>

inline constexpr double kDefaultReconstructionTolerance = 1.0e-8;
inline constexpr long double kDefaultLogdetTolerance = 1.0e-8L;

/**
 * @brief Captures the key mathematical metrics from one factorisation run.
 *
 * The helpers fill this structure when callers request the measured values.
 */
struct FactorisationMetrics
{
    double elapsed_seconds = 0.0;
    LogDetValue logdet_factor = 0.0L;
    double reconstruction_error = 0.0;
};

/**
 * @brief Hold the captured output from one subprocess-based test command.
 *
 * The helper layer uses this structure for CLI, CSV, and plotting smoke tests.
 */
struct CommandResult
{
    int exit_code = -1;
    std::string stdout_text;
    std::string stderr_text;
};

/**
 * @brief Hold parsed CSV header and row dictionaries.
 *
 * Each row maps column names to field values from the same line.
 */
struct CsvTable
{
    std::vector<std::string> header;
    std::vector<std::map<std::string, std::string>> rows;
};

/**
 * @brief Create and clean up a unique temporary directory for one test.
 *
 * The directory is removed recursively when the object goes out of scope.
 */
class ScopedTemporaryDirectory
{
public:
    /**
     * @brief Create a unique temporary directory with the given prefix.
     *
     * @param prefix Prefix used when naming the directory.
     */
    explicit ScopedTemporaryDirectory(const std::string& prefix);

    /**
     * @brief Remove the temporary directory and its contents.
     */
    ~ScopedTemporaryDirectory();

    ScopedTemporaryDirectory(const ScopedTemporaryDirectory&) = delete;
    ScopedTemporaryDirectory& operator=(const ScopedTemporaryDirectory&) = delete;

    /**
     * @brief Return the filesystem path for this temporary directory.
     *
     * @return Temporary directory path.
     */
    const std::filesystem::path& path() const;

private:
    std::filesystem::path directory_;
};

/**
 * @brief Compare two scalars with combined absolute and relative tolerances.
 * @param a First value.
 * @param b Second value.
 * @param atol Absolute tolerance.
 * @param rtol Relative tolerance.
 * @return `true` when the values are close enough.
 */
bool nearly_equal(double a, double b, double atol = 1e-12, double rtol = 1e-12);

/**
 * @brief Compare two vectors element-wise with scalar tolerances.
 * @param a First vector.
 * @param b Second vector.
 * @param atol Absolute tolerance.
 * @param rtol Relative tolerance.
 * @return `true` when the vectors have the same size and all elements are close.
 */
bool vectors_close(const std::vector<double>& a,
                   const std::vector<double>& b,
                   double atol = 1e-12,
                   double rtol = 1e-12);

/**
 * @brief Check whether a dense matrix is symmetric within tolerance.
 * @param a Row-major matrix storage.
 * @param n Matrix dimension.
 * @param atol Absolute tolerance.
 * @param rtol Relative tolerance.
 * @return `true` when the mirrored entries agree within tolerance.
 */
bool matrix_is_symmetric(const std::vector<double>& a,
                         int n,
                         double atol = 1e-12,
                         double rtol = 1e-12);

/**
 * @brief Check whether all diagonal entries are strictly positive.
 * @param a Row-major matrix storage.
 * @param n Matrix dimension.
 * @return `true` when every diagonal entry is greater than zero.
 */
bool diagonal_is_positive(const std::vector<double>& a, int n);

/**
 * @brief Print a dense matrix for debugging.
 * @param a Row-major matrix storage.
 * @param n Matrix dimension.
 */
void print_matrix(const std::vector<double>& a, int n);

/**
 * @brief Build an identity matrix in row-major storage.
 * @param n Matrix dimension.
 * @return Dense identity matrix.
 */
std::vector<double> make_identity_matrix(int n);

/**
 * @brief Build a diagonal matrix from the supplied diagonal values.
 * @param diag Diagonal entries.
 * @return Dense row-major diagonal matrix.
 */
std::vector<double> make_diagonal_matrix(const std::vector<double>& diag);

/**
 * @brief Return the small worked example used in the coursework brief.
 * @return Dense `2 x 2` SPD matrix in row-major storage.
 */
std::vector<double> make_brief_example_matrix();

/**
 * @brief Extract the lower-triangular factor from mirrored factor storage.
 * @param c Mirrored factorised matrix.
 * @param n Matrix dimension.
 * @return Lower-triangular factor with zeros above the diagonal.
 */
std::vector<double> lower_factor_from_storage(const std::vector<double>& c, int n);

/**
 * @brief Return all implementations covered by the broader correctness tests.
 * @return Ordered list of tested implementations.
 */
const std::vector<CholeskyVersion>& all_test_versions();

/**
 * @brief Return the OpenMP implementations covered by the test suite.
 * @return Ordered list of OpenMP implementations.
 */
const std::vector<CholeskyVersion>& openmp_test_versions();

/**
 * @brief Return the blocked implementations covered by the test suite.
 * @return Ordered list of blocked implementations.
 */
const std::vector<CholeskyVersion>& blocked_test_versions();

/**
 * @brief Return the blocked OpenMP implementations covered by the test suite.
 * @return Ordered list of blocked OpenMP implementations.
 */
const std::vector<CholeskyVersion>& blocked_openmp_test_versions();

/**
 * @brief Check whether a version belongs to the OpenMP test set.
 * @param version Implementation identifier.
 * @return `true` when the version is an OpenMP implementation under test.
 */
bool is_openmp_test_version(CholeskyVersion version);

/**
 * @brief Check whether a version belongs to the blocked test set.
 * @param version Implementation identifier.
 * @return `true` when the version accepts a block size.
 */
bool is_blocked_test_version(CholeskyVersion version);

/**
 * @brief Filter requested thread counts to those supported by the current environment.
 * @param requested_counts Requested OpenMP thread counts.
 * @return Supported thread counts, always including `1`.
 */
std::vector<int> supported_thread_counts(const std::vector<int>& requested_counts);

/**
 * @brief Set the OpenMP thread count used by the correctness tests.
 * @param thread_count Requested thread count.
 */
void set_test_openmp_thread_count(int thread_count);

/**
 * @brief Compare two long-double values with combined tolerances.
 * @param a First value.
 * @param b Second value.
 * @param atol Absolute tolerance.
 * @param rtol Relative tolerance.
 * @return `true` when the values are close enough.
 */
bool long_double_nearly_equal(LogDetValue a,
                              LogDetValue b,
                              LogDetValue atol = kDefaultLogdetTolerance,
                              LogDetValue rtol = kDefaultLogdetTolerance);

/**
 * @brief Check whether a matrix is symmetric within tolerance.
 * @param matrix Row-major matrix storage.
 * @param n Matrix dimension.
 * @param atol Absolute tolerance.
 * @param rtol Relative tolerance.
 * @return `true` when the matrix shape and mirrored entries are valid.
 */
bool matrix_is_symmetric_within(const std::vector<double>& matrix,
                                int n,
                                double atol = 1.0e-10,
                                double rtol = 1.0e-10);

/**
 * @brief Check whether all diagonal entries are strictly positive.
 * @param matrix Row-major matrix storage.
 * @param n Matrix dimension.
 * @return `true` when every diagonal entry is greater than zero.
 */
bool diagonal_entries_positive(const std::vector<double>& matrix, int n);

/**
 * @brief Check whether all matrix entries are finite.
 * @param matrix Row-major matrix storage.
 * @return `true` when the matrix contains no NaN or Inf values.
 */
bool matrix_all_finite(const std::vector<double>& matrix);

/**
 * @brief Build a trusted log-determinant reference for one matrix.
 * @param original Source SPD matrix.
 * @param n Matrix dimension.
 * @param logdet Output reference log-determinant.
 * @return `true` when a reference value was produced.
 */
bool reference_logdet_for_matrix(const std::vector<double>& original, int n, LogDetValue& logdet);

/**
 * @brief Run one implementation with its default configuration.
 * @param matrix Matrix storage modified in place.
 * @param n Matrix dimension.
 * @param version Implementation identifier.
 * @return Elapsed time in seconds, or a negative error code.
 */
double run_version_default(std::vector<double>& matrix, int n, CholeskyVersion version);

/**
 * @brief Run one implementation with an explicit block size when supported.
 * @param matrix Matrix storage modified in place.
 * @param n Matrix dimension.
 * @param version Implementation identifier.
 * @param block_size Requested block size.
 * @return Elapsed time in seconds, or a negative error code.
 */
double run_version_with_block_size(std::vector<double>& matrix,
                                   int n,
                                   CholeskyVersion version,
                                   int block_size);

/**
 * @brief Check that a factorised matrix satisfies the mathematical correctness conditions.
 * @param original Original SPD matrix.
 * @param factorised Factorised matrix storage.
 * @param n Matrix dimension.
 * @param reference_logdet Trusted log-determinant reference.
 * @param failure Output failure message when validation fails.
 * @param metrics Optional output metrics structure.
 * @param reconstruction_tolerance Maximum allowed reconstruction error.
 * @param logdet_tolerance Maximum allowed log-determinant difference.
 * @return `true` when all validation checks pass.
 *
 * @note The factor is expected to be mirrored into full row-major storage.
 */
bool validate_factorisation(const std::vector<double>& original,
                            const std::vector<double>& factorised,
                            int n,
                            LogDetValue reference_logdet,
                            std::string& failure,
                            FactorisationMetrics* metrics = nullptr,
                            double reconstruction_tolerance =
                                kDefaultReconstructionTolerance,
                            LogDetValue logdet_tolerance = kDefaultLogdetTolerance);

/**
 * @brief Return the repository source root compiled into the test binaries.
 *
 * @return Absolute source-root path.
 */
std::filesystem::path coursework_source_root();

/**
 * @brief Return the build root compiled into the test binaries.
 *
 * @return Absolute build-root path.
 */
std::filesystem::path coursework_binary_root();

/**
 * @brief Return the benchmark executable used by subprocess-driven tests.
 *
 * @return Absolute path to `run_cholesky`.
 */
std::filesystem::path benchmark_executable_path();

/**
 * @brief Return the plotting CLI script used by plotting smoke tests.
 *
 * @return Absolute path to `plot/cholesky_plotter.py`.
 */
std::filesystem::path plotter_script_path();

/**
 * @brief Return the Python interpreter compiled into the test binaries.
 *
 * @return Interpreter path or executable name.
 */
std::string python_executable_path();

/**
 * @brief Run one subprocess and capture its stdout, stderr, and exit code.
 *
 * @param args Command-line arguments, including the executable.
 * @param cwd Optional working directory for the command.
 * @param env_overrides Optional extra environment variables.
 * @return Captured command result.
 */
CommandResult run_command(const std::vector<std::string>& args,
                          const std::filesystem::path& cwd = {},
                          const std::vector<std::pair<std::string, std::string>>& env_overrides = {});

/**
 * @brief Run the benchmark executable with the supplied mode arguments.
 *
 * @param args Benchmark arguments after `run_cholesky`.
 * @param cwd Optional working directory for the command.
 * @param env_overrides Optional extra environment variables.
 * @return Captured command result.
 */
CommandResult run_benchmark(const std::vector<std::string>& args,
                            const std::filesystem::path& cwd = {},
                            const std::vector<std::pair<std::string, std::string>>& env_overrides = {});

/**
 * @brief Read a text file as raw lines without trailing newline characters.
 *
 * @param path File path.
 * @return File contents split into lines.
 */
std::vector<std::string> read_text_lines(const std::filesystem::path& path);

/**
 * @brief Parse a simple benchmark CSV into header and row dictionaries.
 *
 * @param path CSV file path.
 * @return Parsed CSV data.
 */
CsvTable read_csv_table(const std::filesystem::path& path);

/**
 * @brief Return the summary CSV path for a raw benchmark CSV.
 *
 * @param raw_csv Raw benchmark CSV path.
 * @return Companion `_summary.csv` path.
 */
std::filesystem::path summary_csv_path(const std::filesystem::path& raw_csv);

#endif
