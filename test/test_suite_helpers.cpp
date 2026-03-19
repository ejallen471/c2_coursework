/**
 * @file test_suite_helpers.cpp
 * @brief Shared implementations for the broader correctness and integration-oriented C++ tests.
 */

#include "test_suite_helpers.h"

#include "runtime_cholesky.h"
#include "timer.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <system_error>
#include <utility>
#include <vector>

#include <sys/wait.h>
#include <unistd.h>

// Include the OpenMP runtime only when the test binary was built with OpenMP support so the
// helper layer still compiles on toolchains where OpenMP is unavailable.
#ifdef _OPENMP
#include <omp.h>
#endif

namespace
{
/**
 * @brief Measure the runtime of one test helper callable.
 *
 * @param factorisation Callable that performs the factorisation.
 * @return Elapsed wall-clock time in seconds.
 */
template <typename Factorisation>
double time_factorisation(Factorisation factorisation)
{
    const double t0 = wall_time_seconds();
    factorisation();
    return wall_time_seconds() - t0;
}

/**
 * @brief Quote one shell argument safely for POSIX subprocess helpers.
 *
 * @param text Raw argument text.
 * @return Shell-quoted argument.
 */
std::string shell_quote(const std::string& text)
{
    std::string quoted = "'";

    for (const char ch : text)
    {
        if (ch == '\'')
        {
            quoted += "'\"'\"'";
        }
        else
        {
            quoted.push_back(ch);
        }
    }

    quoted.push_back('\'');
    return quoted;
}

/**
 * @brief Decode a `std::system` status into a process-style exit code.
 *
 * @param raw_status Raw status returned by `std::system`.
 * @return Exit code or signal-style fallback.
 */
int decode_exit_status(int raw_status)
{
    if (raw_status == -1)
    {
        return -1;
    }

    if (WIFEXITED(raw_status))
    {
        return WEXITSTATUS(raw_status);
    }

    if (WIFSIGNALED(raw_status))
    {
        return 128 + WTERMSIG(raw_status);
    }

    return raw_status;
}

/**
 * @brief Split a simple CSV line while preserving trailing empty fields.
 *
 * @param line CSV line with no quoted commas.
 * @return Parsed field values.
 */
std::vector<std::string> split_csv_line(const std::string& line)
{
    std::vector<std::string> fields;
    std::string current;

    for (const char ch : line)
    {
        if (ch == ',')
        {
            fields.push_back(current);
            current.clear();
            continue;
        }

        current.push_back(ch);
    }

    fields.push_back(current);
    return fields;
}
} // namespace

/**
 * @brief Compare two scalars with combined tolerances.
 *
 * @param a First value.
 * @param b Second value.
 * @param atol Absolute tolerance.
 * @param rtol Relative tolerance.
 * @return `true` when the values are close enough.
 */
bool nearly_equal(double a, double b, double atol, double rtol)
{
    const double diff = std::fabs(a - b);
    const double scale = std::max(std::fabs(a), std::fabs(b));
    return diff <= atol + rtol * scale;
}

/**
 * @brief Compare two vectors element-wise with scalar tolerances.
 *
 * @param a First vector.
 * @param b Second vector.
 * @param atol Absolute tolerance.
 * @param rtol Relative tolerance.
 * @return `true` when the vectors have the same size and all elements are close.
 */
bool vectors_close(const std::vector<double>& a, const std::vector<double>& b, double atol, double rtol)
{
    if (a.size() != b.size())
    {
        return false;
    }

    for (std::size_t i = 0; i < a.size(); ++i)
    {
        if (!nearly_equal(a[i], b[i], atol, rtol))
        {
            return false;
        }
    }

    return true;
}

/**
 * @brief Forward matrix symmetry checks to the stricter shared helper.
 *
 * @param a Row-major matrix storage.
 * @param n Matrix dimension.
 * @param atol Absolute tolerance.
 * @param rtol Relative tolerance.
 * @return `true` when the matrix is symmetric within tolerance.
 */
bool matrix_is_symmetric(const std::vector<double>& a, int n, double atol, double rtol)
{
    return matrix_is_symmetric_within(a, n, atol, rtol);
}

/**
 * @brief Forward diagonal positivity checks to the stricter shared helper.
 *
 * @param a Row-major matrix storage.
 * @param n Matrix dimension.
 * @return `true` when every diagonal entry is strictly positive.
 */
bool diagonal_is_positive(const std::vector<double>& a, int n)
{
    return diagonal_entries_positive(a, n);
}

/**
 * @brief Print a dense matrix for debugging failed tests.
 *
 * @param a Row-major matrix storage.
 * @param n Matrix dimension.
 */
void print_matrix(const std::vector<double>& a, int n)
{
    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            std::cout << a[static_cast<std::size_t>(i) * static_cast<std::size_t>(n) +
                           static_cast<std::size_t>(j)]
                      << ' ';
        }
        std::cout << '\n';
    }
}

/**
 * @brief Build an identity matrix for deterministic test cases.
 *
 * @param n Matrix dimension.
 * @return Dense row-major identity matrix.
 */
std::vector<double> make_identity_matrix(int n)
{
    std::vector<double> matrix(static_cast<std::size_t>(n) * static_cast<std::size_t>(n), 0.0);

    for (int i = 0; i < n; ++i)
    {
        matrix[static_cast<std::size_t>(i) * static_cast<std::size_t>(n) +
               static_cast<std::size_t>(i)] = 1.0;
    }

    return matrix;
}

/**
 * @brief Build a diagonal matrix from supplied diagonal entries.
 *
 * @param diag Diagonal entries.
 * @return Dense row-major diagonal matrix.
 */
std::vector<double> make_diagonal_matrix(const std::vector<double>& diag)
{
    const int n = static_cast<int>(diag.size());
    std::vector<double> matrix(static_cast<std::size_t>(n) * static_cast<std::size_t>(n), 0.0);

    for (int i = 0; i < n; ++i)
    {
        matrix[static_cast<std::size_t>(i) * static_cast<std::size_t>(n) +
               static_cast<std::size_t>(i)] = diag[static_cast<std::size_t>(i)];
    }

    return matrix;
}

/**
 * @brief Return the small `2 x 2` SPD example used in unit tests.
 *
 * @return Dense row-major matrix.
 */
std::vector<double> make_brief_example_matrix()
{
    return {4.0, 2.0, 2.0, 26.0};
}

/**
 * @brief Extract the lower factor from mirrored factor storage.
 *
 * @param c Mirrored factorised matrix.
 * @param n Matrix dimension.
 * @return Lower-triangular factor with zeros above the diagonal.
 */
std::vector<double> lower_factor_from_storage(const std::vector<double>& c, int n)
{
    std::vector<double> lower(static_cast<std::size_t>(n) * static_cast<std::size_t>(n), 0.0);

    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j <= i; ++j)
        {
            lower[static_cast<std::size_t>(i) * static_cast<std::size_t>(n) +
                  static_cast<std::size_t>(j)] =
                c[static_cast<std::size_t>(i) * static_cast<std::size_t>(n) +
                  static_cast<std::size_t>(j)];
        }
    }

    return lower;
}

/**
 * @brief Return all implementations covered by the correctness tests.
 *
 * @return Ordered list of tested implementations.
 */
const std::vector<CholeskyVersion>& all_test_versions()
{
    static const std::vector<CholeskyVersion> versions = {
        CholeskyVersion::Baseline,
        CholeskyVersion::LowerTriangleOnly,
        CholeskyVersion::UpperTriangle,
        CholeskyVersion::ContiguousAccess,
        CholeskyVersion::BlockedTileKernels,
        CholeskyVersion::BlockedTileKernelsUnrolled,
        CholeskyVersion::OpenMPRowParallelUnblocked,
        CholeskyVersion::OpenMPTileParallelBlocked,
        CholeskyVersion::OpenMPBlockRowParallel,
        CholeskyVersion::OpenMPTileListParallel,
        CholeskyVersion::OpenMPTaskDAGBlocked,
    };

    return versions;
}

/**
 * @brief Return the OpenMP implementations covered by the tests.
 *
 * @return Ordered list of OpenMP implementations.
 */
const std::vector<CholeskyVersion>& openmp_test_versions()
{
    static const std::vector<CholeskyVersion> versions = {
        CholeskyVersion::OpenMPRowParallelUnblocked,
        CholeskyVersion::OpenMPTileParallelBlocked,
        CholeskyVersion::OpenMPBlockRowParallel,
        CholeskyVersion::OpenMPTileListParallel,
        CholeskyVersion::OpenMPTaskDAGBlocked,
    };

    return versions;
}

/**
 * @brief Return the blocked implementations covered by the tests.
 *
 * @return Ordered list of blocked implementations.
 */
const std::vector<CholeskyVersion>& blocked_test_versions()
{
    static const std::vector<CholeskyVersion> versions = {
        CholeskyVersion::BlockedTileKernels,
        CholeskyVersion::BlockedTileKernelsUnrolled,
        CholeskyVersion::OpenMPTileParallelBlocked,
        CholeskyVersion::OpenMPBlockRowParallel,
        CholeskyVersion::OpenMPTileListParallel,
        CholeskyVersion::OpenMPTaskDAGBlocked,
    };

    return versions;
}

/**
 * @brief Return the blocked OpenMP implementations covered by the tests.
 *
 * @return Ordered list of blocked OpenMP implementations.
 */
const std::vector<CholeskyVersion>& blocked_openmp_test_versions()
{
    static const std::vector<CholeskyVersion> versions = {
        CholeskyVersion::OpenMPTileParallelBlocked,
        CholeskyVersion::OpenMPBlockRowParallel,
        CholeskyVersion::OpenMPTileListParallel,
        CholeskyVersion::OpenMPTaskDAGBlocked,
    };

    return versions;
}

/**
 * @brief Check whether a version belongs to the OpenMP test set.
 *
 * @param version Implementation identifier.
 * @return `true` when the version is an OpenMP implementation under test.
 */
bool is_openmp_test_version(CholeskyVersion version)
{
    return std::find(
               openmp_test_versions().begin(), openmp_test_versions().end(), version) !=
        openmp_test_versions().end();
}

/**
 * @brief Check whether a version belongs to the blocked test set.
 *
 * @param version Implementation identifier.
 * @return `true` when the version accepts an explicit block size.
 */
bool is_blocked_test_version(CholeskyVersion version)
{
    return std::find(
               blocked_test_versions().begin(), blocked_test_versions().end(), version) !=
        blocked_test_versions().end();
}

/**
 * @brief Filter requested thread counts to those supported by the current runtime.
 *
 * @param requested_counts Requested OpenMP thread counts.
 * @return Supported thread counts, always including `1`.
 */
std::vector<int> supported_thread_counts(const std::vector<int>& requested_counts)
{
    int available_threads = 1;
#if defined(_OPENMP)
    // Query the OpenMP runtime for the current upper bound so the tests only request thread
    // counts that the executing environment can actually provide.
    available_threads = std::max(1, omp_get_max_threads());
#endif

    std::vector<int> supported;
    supported.push_back(1);

    for (const int requested : requested_counts)
    {
        if (requested <= 1 || requested > available_threads)
        {
            continue;
        }

        if (std::find(supported.begin(), supported.end(), requested) == supported.end())
        {
            supported.push_back(requested);
        }
    }

    return supported;
}

/**
 * @brief Set the OpenMP thread count used by test runs.
 *
 * @param thread_count Requested thread count.
 */
void set_test_openmp_thread_count(int thread_count)
{
    const int safe_thread_count = std::max(1, thread_count);

#if defined(_OPENMP)
    // Push the requested test thread count into the OpenMP runtime so each correctness run
    // exercises the intended degree of parallelism rather than a stale inherited setting.
    omp_set_num_threads(safe_thread_count);
#endif
}

/**
 * @brief Compare two long-double values with combined tolerances.
 *
 * @param a First value.
 * @param b Second value.
 * @param atol Absolute tolerance.
 * @param rtol Relative tolerance.
 * @return `true` when the values are close enough.
 */
bool long_double_nearly_equal(LogDetValue a, LogDetValue b, LogDetValue atol, LogDetValue rtol)
{
    const LogDetValue diff = std::fabs(a - b);
    const LogDetValue scale = std::max(std::fabs(a), std::fabs(b));
    return diff <= atol + rtol * scale;
}

/**
 * @brief Check whether a row-major matrix is symmetric within tolerance.
 *
 * @param matrix Row-major matrix storage.
 * @param n Matrix dimension.
 * @param atol Absolute tolerance.
 * @param rtol Relative tolerance.
 * @return `true` when the matrix shape and mirrored entries are valid.
 */
bool matrix_is_symmetric_within(const std::vector<double>& matrix, int n, double atol, double rtol)
{
    if (matrix.size() != static_cast<std::size_t>(n) * static_cast<std::size_t>(n))
    {
        return false;
    }

    for (int i = 0; i < n; ++i)
    {
        for (int j = i + 1; j < n; ++j)
        {
            const double a = matrix[static_cast<std::size_t>(i) * static_cast<std::size_t>(n) +
                                    static_cast<std::size_t>(j)];
            const double b = matrix[static_cast<std::size_t>(j) * static_cast<std::size_t>(n) +
                                    static_cast<std::size_t>(i)];
            const double diff = std::fabs(a - b);
            const double scale = std::max(std::fabs(a), std::fabs(b));

            if (diff > atol + rtol * scale)
            {
                return false;
            }
        }
    }

    return true;
}

/**
 * @brief Check whether all diagonal entries are strictly positive.
 *
 * @param matrix Row-major matrix storage.
 * @param n Matrix dimension.
 * @return `true` when every diagonal entry is greater than zero.
 */
bool diagonal_entries_positive(const std::vector<double>& matrix, int n)
{
    if (matrix.size() != static_cast<std::size_t>(n) * static_cast<std::size_t>(n))
    {
        return false;
    }

    for (int i = 0; i < n; ++i)
    {
        const std::size_t index =
            static_cast<std::size_t>(i) * static_cast<std::size_t>(n) +
            static_cast<std::size_t>(i);
        if (!(matrix[index] > 0.0))
        {
            return false;
        }
    }

    return true;
}

/**
 * @brief Check whether all matrix entries are finite.
 *
 * @param matrix Row-major matrix storage.
 * @return `true` when the matrix contains no NaN or Inf values.
 */
bool matrix_all_finite(const std::vector<double>& matrix)
{
    return std::all_of(matrix.begin(),
                       matrix.end(),
                       [](double value) { return std::isfinite(value) != 0; });
}

/**
 * @brief Build a trusted log-determinant reference for one matrix.
 *
 * @param original Source SPD matrix.
 * @param n Matrix dimension.
 * @param logdet Output reference log-determinant.
 * @return `true` when a reference value was produced.
 */
bool reference_logdet_for_matrix(const std::vector<double>& original, int n, LogDetValue& logdet)
{
    return lapack_reference_logdet(original, n, logdet);
}

/**
 * @brief Run one implementation with its default runtime configuration.
 *
 * @param matrix Matrix storage modified in place.
 * @param n Matrix dimension.
 * @param version Implementation identifier.
 * @return Elapsed time in seconds, or a negative error code.
 */
double run_version_default(std::vector<double>& matrix, int n, CholeskyVersion version)
{
    return run_cholesky_version(matrix.data(), n, version);
}

/**
 * @brief Run one implementation with an explicit block size when supported.
 *
 * @param matrix Matrix storage modified in place.
 * @param n Matrix dimension.
 * @param version Implementation identifier.
 * @param block_size Requested block size.
 * @return Elapsed time in seconds, or a negative error code.
 */
double run_version_with_block_size(std::vector<double>& matrix,
                                   int n,
                                   CholeskyVersion version,
                                   int block_size)
{
    if (!is_blocked_test_version(version))
    {
        return run_version_default(matrix, n, version);
    }

    const std::size_t matrix_size = static_cast<std::size_t>(n);
    const std::size_t block_size_value = static_cast<std::size_t>(std::max(1, block_size));

    switch (version)
    {
    case CholeskyVersion::BlockedTileKernels:
        return time_factorisation(
            [&]() { cholesky_blocked_tile_kernels(matrix.data(), matrix_size, block_size_value); });

    case CholeskyVersion::BlockedTileKernelsUnrolled:
        return time_factorisation(
            [&]() {
                cholesky_blocked_tile_kernels_unrolled(
                    matrix.data(), matrix_size, block_size_value);
            });

    case CholeskyVersion::OpenMPTileParallelBlocked:
        return time_factorisation([&]() {
            cholesky_openmp_tile_parallel_blocked(matrix.data(), matrix_size, block_size_value);
        });

    case CholeskyVersion::OpenMPBlockRowParallel:
        return time_factorisation([&]() {
            cholesky_openmp_block_row_parallel(matrix.data(), matrix_size, block_size_value);
        });

    case CholeskyVersion::OpenMPTileListParallel:
        return time_factorisation([&]() {
            cholesky_openmp_tile_list_parallel(matrix.data(), matrix_size, block_size_value);
        });

    case CholeskyVersion::OpenMPTaskDAGBlocked:
        return time_factorisation([&]() {
            cholesky_openmp_task_dag_blocked(matrix.data(), matrix_size, block_size_value);
        });

    default:
        return run_version_default(matrix, n, version);
    }
}

/**
 * @brief Check that a factorised matrix satisfies the mathematical correctness rules.
 *
 * @param original Original SPD matrix.
 * @param factorised Factorised matrix storage.
 * @param n Matrix dimension.
 * @param reference_logdet Trusted log-determinant reference.
 * @param failure Output failure message when validation fails.
 * @param metrics Optional output metrics structure.
 * @param reconstruction_tolerance Maximum allowed reconstruction error.
 * @param logdet_tolerance Maximum allowed log-determinant difference.
 * @return `true` when all validation checks pass.
 */
bool validate_factorisation(const std::vector<double>& original,
                            const std::vector<double>& factorised,
                            int n,
                            LogDetValue reference_logdet,
                            std::string& failure,
                            FactorisationMetrics* metrics,
                            double reconstruction_tolerance,
                            LogDetValue logdet_tolerance)
{
    if (factorised.size() != original.size())
    {
        failure = "factorised matrix changed shape";
        return false;
    }

    if (!matrix_all_finite(factorised))
    {
        failure = "factorised matrix contains NaN or Inf";
        return false;
    }

    if (!matrix_is_symmetric_within(factorised, n))
    {
        failure = "factorised matrix is not mirrored symmetrically";
        return false;
    }

    if (!diagonal_entries_positive(factorised, n))
    {
        failure = "factor diagonal is not strictly positive";
        return false;
    }

    const std::vector<double> reconstructed =
        reconstruct_from_factorised_storage(factorised, static_cast<std::size_t>(n));
    const double reconstruction_error = relative_frobenius_error(reconstructed, original);
    if (metrics != nullptr)
    {
        metrics->reconstruction_error = reconstruction_error;
    }

    if (!(reconstruction_error <= reconstruction_tolerance))
    {
        std::ostringstream message;
        message << "relative reconstruction error " << reconstruction_error
                << " exceeded tolerance " << reconstruction_tolerance;
        failure = message.str();
        return false;
    }

    const LogDetValue logdet_factor =
        logdet_from_factorised_storage(factorised, static_cast<std::size_t>(n));
    if (metrics != nullptr)
    {
        metrics->logdet_factor = logdet_factor;
    }

    if (!long_double_nearly_equal(
            logdet_factor, reference_logdet, logdet_tolerance, logdet_tolerance))
    {
        std::ostringstream message;
        message << "logdet mismatch: factor=" << static_cast<double>(logdet_factor)
                << " reference=" << static_cast<double>(reference_logdet);
        failure = message.str();
        return false;
    }

    return true;
}

/**
 * @brief Create a unique temporary directory for one test scope.
 *
 * @param prefix Prefix used when naming the directory.
 */
ScopedTemporaryDirectory::ScopedTemporaryDirectory(const std::string& prefix)
{
    const std::filesystem::path template_path =
        std::filesystem::temp_directory_path() / (prefix + "-XXXXXX");
    std::string mutable_template = template_path.string();
    std::vector<char> buffer(mutable_template.begin(), mutable_template.end());
    buffer.push_back('\0');

    char* created_path = ::mkdtemp(buffer.data());
    if (created_path == nullptr)
    {
        throw std::runtime_error("failed to create temporary directory");
    }

    directory_ = std::filesystem::path(created_path);
}

/**
 * @brief Remove the temporary directory recursively.
 */
ScopedTemporaryDirectory::~ScopedTemporaryDirectory()
{
    if (!directory_.empty())
    {
        std::error_code error;
        std::filesystem::remove_all(directory_, error);
    }
}

/**
 * @brief Return this scope's temporary directory path.
 *
 * @return Temporary directory path.
 */
const std::filesystem::path& ScopedTemporaryDirectory::path() const
{
    return directory_;
}

/**
 * @brief Return the compiled-in repository source root.
 *
 * @return Absolute source-root path.
 */
std::filesystem::path coursework_source_root()
{
    return std::filesystem::path(COURSEWORK_SOURCE_ROOT);
}

/**
 * @brief Return the compiled-in build root.
 *
 * @return Absolute build-root path.
 */
std::filesystem::path coursework_binary_root()
{
    return std::filesystem::path(COURSEWORK_BINARY_ROOT);
}

/**
 * @brief Return the benchmark executable path used by subprocess tests.
 *
 * @return Absolute path to `run_cholesky`.
 */
std::filesystem::path benchmark_executable_path()
{
    return coursework_binary_root() / "run" / "run_cholesky";
}

/**
 * @brief Return the plotting CLI path used by plotting smoke tests.
 *
 * @return Absolute path to `plot/cholesky_plotter.py`.
 */
std::filesystem::path plotter_script_path()
{
    return coursework_source_root() / "plot" / "cholesky_plotter.py";
}

/**
 * @brief Return the Python interpreter path compiled into the test binaries.
 *
 * @return Interpreter path or fallback executable name.
 */
std::string python_executable_path()
{
#ifdef COURSEWORK_PYTHON_EXECUTABLE
    return COURSEWORK_PYTHON_EXECUTABLE;
#else
    return "python3";
#endif
}

/**
 * @brief Run one subprocess and capture stdout and stderr into temporary files.
 *
 * @param args Command-line arguments, including the executable.
 * @param cwd Optional working directory.
 * @param env_overrides Optional extra environment variables.
 * @return Captured command result.
 */
CommandResult run_command(const std::vector<std::string>& args,
                          const std::filesystem::path& cwd,
                          const std::vector<std::pair<std::string, std::string>>& env_overrides)
{
    if (args.empty())
    {
        throw std::invalid_argument("run_command requires at least one argument");
    }

    ScopedTemporaryDirectory temp_dir("coursework-command");
    const std::filesystem::path stdout_path = temp_dir.path() / "stdout.txt";
    const std::filesystem::path stderr_path = temp_dir.path() / "stderr.txt";
    const std::filesystem::path mplconfig_path = temp_dir.path() / "mplconfig";
    const std::filesystem::path cache_path = temp_dir.path() / "cache";
    std::filesystem::create_directories(mplconfig_path);
    std::filesystem::create_directories(cache_path);

    std::ostringstream command;
    if (!cwd.empty())
    {
        command << "(cd " << shell_quote(cwd.string()) << " && ";
    }

    command << "MPLCONFIGDIR=" << shell_quote(mplconfig_path.string()) << ' ';
    command << "XDG_CACHE_HOME=" << shell_quote(cache_path.string()) << ' ';
    for (const auto& [name, value] : env_overrides)
    {
        command << name << "=" << shell_quote(value) << ' ';
    }

    for (std::size_t i = 0; i < args.size(); ++i)
    {
        if (i != 0U)
        {
            command << ' ';
        }

        command << shell_quote(args[i]);
    }

    command << " >" << shell_quote(stdout_path.string()) << " 2>"
            << shell_quote(stderr_path.string());

    if (!cwd.empty())
    {
        command << ')';
    }

    const int raw_status = std::system(command.str().c_str());

    CommandResult result;
    result.exit_code = decode_exit_status(raw_status);

    {
        std::ifstream stdout_stream(stdout_path);
        std::ostringstream buffer;
        buffer << stdout_stream.rdbuf();
        result.stdout_text = buffer.str();
    }

    {
        std::ifstream stderr_stream(stderr_path);
        std::ostringstream buffer;
        buffer << stderr_stream.rdbuf();
        result.stderr_text = buffer.str();
    }

    return result;
}

/**
 * @brief Run the benchmark executable with the supplied arguments.
 *
 * @param args Benchmark arguments after `run_cholesky`.
 * @param cwd Optional working directory.
 * @param env_overrides Optional extra environment variables.
 * @return Captured command result.
 */
CommandResult run_benchmark(const std::vector<std::string>& args,
                            const std::filesystem::path& cwd,
                            const std::vector<std::pair<std::string, std::string>>& env_overrides)
{
    std::vector<std::string> command;
    command.reserve(args.size() + 1U);
    command.push_back(benchmark_executable_path().string());
    command.insert(command.end(), args.begin(), args.end());
    return run_command(command, cwd, env_overrides);
}

/**
 * @brief Read a text file as raw lines without trailing newline characters.
 *
 * @param path File path.
 * @return File contents split into lines.
 */
std::vector<std::string> read_text_lines(const std::filesystem::path& path)
{
    std::vector<std::string> lines;
    std::ifstream input(path);
    std::string line;

    while (std::getline(input, line))
    {
        lines.push_back(line);
    }

    return lines;
}

/**
 * @brief Parse a benchmark CSV into header and row dictionaries.
 *
 * @param path CSV file path.
 * @return Parsed CSV data.
 */
CsvTable read_csv_table(const std::filesystem::path& path)
{
    CsvTable table;
    const std::vector<std::string> lines = read_text_lines(path);
    if (lines.empty())
    {
        return table;
    }

    table.header = split_csv_line(lines.front());
    for (std::size_t line_index = 1; line_index < lines.size(); ++line_index)
    {
        if (lines[line_index].empty())
        {
            continue;
        }

        const std::vector<std::string> fields = split_csv_line(lines[line_index]);
        std::map<std::string, std::string> row;

        for (std::size_t column_index = 0; column_index < table.header.size(); ++column_index)
        {
            const std::string value =
                column_index < fields.size() ? fields[column_index] : std::string();
            row.emplace(table.header[column_index], value);
        }

        table.rows.push_back(std::move(row));
    }

    return table;
}

/**
 * @brief Return the companion summary CSV path for a raw benchmark CSV.
 *
 * @param raw_csv Raw benchmark CSV path.
 * @return Summary CSV path with the `_summary.csv` suffix.
 */
std::filesystem::path summary_csv_path(const std::filesystem::path& raw_csv)
{
    return raw_csv.parent_path() / (raw_csv.stem().string() + "_summary.csv");
}
