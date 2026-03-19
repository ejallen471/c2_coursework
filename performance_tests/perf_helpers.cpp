/**
 * @file perf_helpers.cpp
 * @brief Helper functions for benchmark output, statistics, and correctness checks.
 */

#include "perf_helpers.h"

#include "matrix.h"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <sstream>
#include <system_error>

#if !defined(_WIN32)
#include <unistd.h>
#endif

extern "C"
{
    // LAPACK routine for Cholesky factorisation of a symmetric positive definite matrix.
    void dpotrf_(const char* uplo, const int* n, double* a, const int* lda, int* info);
}

namespace
{
/**
 * @brief Flush a CSV file and force the data to disk where supported.
 *
 * This helps make sure benchmark output is not left sitting in a buffer if the run
 * ends unexpectedly.
 *
 * @param output Open file handle.
 * @return `true` if the flush succeeded.
 */
bool flush_csv_file(FILE* output)
{
    if (std::fflush(output) != 0)
    {
        return false;
    }

#if defined(_WIN32)
    return true;
#else
    const int fd = fileno(output);
    if (fd == -1)
    {
        return false;
    }

    return ::fsync(fd) == 0;
#endif
}

/**
 * @brief Write text to a CSV file using the given file mode.
 *
 * This is used both for creating new CSV files and appending new rows.
 *
 * @param path Output file path.
 * @param text Text to write.
 * @param mode File mode such as `"w"` for overwrite or `"a"` for append.
 * @return `true` if the write and flush both succeed.
 */
bool write_csv_text_with_mode(const std::filesystem::path& path,
                              const std::string& text,
                              const char* mode)
{
    FILE* output = std::fopen(path.string().c_str(), mode);
    if (output == nullptr)
    {
        return false;
    }

    const bool ok = (std::fputs(text.c_str(), output) != EOF) && flush_csv_file(output);
    return std::fclose(output) == 0 && ok;
}
} // namespace

BenchmarkCsvSession::BenchmarkCsvSession(std::filesystem::path raw_csv_path, bool writes_summary)
    : raw_csv_path_(std::move(raw_csv_path)),
      summary_csv_path_(writes_summary ? summary_csv_path_for(raw_csv_path_)
                                       : std::filesystem::path()),
      writes_summary_(writes_summary)
{
}

bool BenchmarkCsvSession::initialise(const std::string& raw_header,
                                     std::ostream& error_output,
                                     const std::string& summary_header) const
{
    if (raw_csv_path_.has_parent_path())
    {
        std::error_code directory_error;
        std::filesystem::create_directories(raw_csv_path_.parent_path(), directory_error);
        if (directory_error)
        {
            error_output << "Error: failed to create output directory "
                         << raw_csv_path_.parent_path() << " (" << directory_error.message()
                         << ")\n";
            return false;
        }
    }

    if (!overwrite_csv_text(raw_csv_path_, raw_header))
    {
        error_output << "Error: failed to write CSV header to " << raw_csv_path_ << " ("
                     << std::strerror(errno) << ")\n";
        return false;
    }

    if (!writes_summary_)
    {
        return true;
    }

    if (summary_header.empty())
    {
        error_output << "Error: missing summary CSV header for " << summary_csv_path_ << '\n';
        return false;
    }

    if (!overwrite_csv_text(summary_csv_path_, summary_header))
    {
        error_output << "Error: failed to write summary CSV header to " << summary_csv_path_ << " ("
                     << std::strerror(errno) << ")\n";
        return false;
    }

    return true;
}

bool BenchmarkCsvSession::append_raw_row(const std::string& row,
                                         std::ostream& error_output,
                                         const std::string& context) const
{
    if (append_csv_text(raw_csv_path_, row))
    {
        return true;
    }

    error_output << "Error: failed to append raw CSV row";
    if (!context.empty())
    {
        error_output << " for " << context;
    }
    error_output << '\n';
    return false;
}

bool BenchmarkCsvSession::append_summary_row(const std::string& row,
                                             std::ostream& error_output,
                                             const std::string& context) const
{
    if (!writes_summary_)
    {
        error_output << "Error: attempted to append a summary row without a summary CSV\n";
        return false;
    }

    if (append_csv_text(summary_csv_path_, row))
    {
        return true;
    }

    error_output << "Error: failed to append summary CSV row";
    if (!context.empty())
    {
        error_output << " for " << context;
    }
    error_output << '\n';
    return false;
}

const std::filesystem::path& BenchmarkCsvSession::raw_csv_path() const
{
    return raw_csv_path_;
}

const std::filesystem::path& BenchmarkCsvSession::summary_csv_path() const
{
    return summary_csv_path_;
}

bool BenchmarkCsvSession::writes_summary() const
{
    return writes_summary_;
}

/**
 * @brief Parse a string as an integer and reject partial or invalid input.
 *
 * Unlike `atoi`, this only accepts a fully valid integer string.
 *
 * @param text Input text.
 * @param value Output integer value on success.
 * @return `true` if parsing succeeds.
 */
bool parse_strict_int(const std::string& text, int& value)
{
    if (text.empty())
    {
        return false;
    }

    char* end = nullptr;
    errno = 0;
    const long parsed = std::strtol(text.c_str(), &end, 10);
    if (end == text.c_str() || *end != '\0' || errno != 0)
    {
        return false;
    }

    if (parsed < static_cast<long>(std::numeric_limits<int>::min()) ||
        parsed > static_cast<long>(std::numeric_limits<int>::max()))
    {
        return false;
    }

    value = static_cast<int>(parsed);
    return true;
}

/**
 * @brief Parse a strictly positive integer.
 *
 * @param text Input text.
 * @param value Output integer value on success.
 * @return `true` if parsing succeeds and the value is greater than zero.
 */
bool parse_positive_int(const std::string& text, int& value)
{
    if (!parse_strict_int(text, value))
    {
        return false;
    }

    return value > 0;
}

/**
 * @brief Compute the log-determinant from a factorised matrix.
 *
 * The matrix is assumed to contain a Cholesky factor on its diagonal, so the
 * determinant is the square of the product of the diagonal entries.
 *
 * @param c Factorised matrix storage.
 * @param n Matrix dimension.
 * @return Log-determinant.
 */
LogDetValue logdet_from_factorised_storage(const std::vector<double>& c, std::size_t n)
{
    LogDetValue sum = 0.0L;

    for (std::size_t i = 0; i < n; ++i)
    {
        const std::size_t index = i * n + i;
        sum += std::log(static_cast<LogDetValue>(c[index]));
    }

    return 2.0L * sum;
}

/**
 * @brief Compute the relative percentage difference between two values.
 *
 * @param value Measured value.
 * @param reference Reference value.
 * @return Relative difference as a percentage.
 */
LogDetValue relative_difference_percent(LogDetValue value, LogDetValue reference)
{
    const LogDetValue scale = std::fabs(reference);

    if (scale == 0.0)
    {
        return (std::fabs(value) == 0.0) ? 0.0L : 100.0L;
    }

    return 100.0L * std::fabs(value - reference) / scale;
}

/**
 * @brief Reconstruct the original matrix from lower-triangular factor storage.
 *
 * This forms `A = L L^T` from the factorised matrix.
 *
 * @param c Factorised matrix storage.
 * @param n Matrix dimension.
 * @return Reconstructed dense matrix.
 */
std::vector<double> reconstruct_from_factorised_storage(const std::vector<double>& c, std::size_t n)
{
    std::vector<double> a(n * n, 0.0);

    for (std::size_t i = 0; i < n; ++i)
    {
        for (std::size_t j = 0; j < n; ++j)
        {
            LogDetValue sum = 0.0L;
            const std::size_t kmax = std::min(i, j);

            for (std::size_t k = 0; k <= kmax; ++k)
            {
                sum +=
                    static_cast<LogDetValue>(c[i * n + k]) * static_cast<LogDetValue>(c[j * n + k]);
            }

            a[i * n + j] = static_cast<double>(sum);
        }
    }

    return a;
}

/**
 * @brief Compute the relative Frobenius norm error between two matrices.
 *
 * @param actual Computed matrix.
 * @param reference Reference matrix.
 * @return Relative Frobenius error, or infinity if the inputs are invalid.
 */
double relative_frobenius_error(const std::vector<double>& actual,
                                const std::vector<double>& reference)
{
    if (actual.size() != reference.size() || actual.empty())
    {
        return std::numeric_limits<double>::infinity();
    }

    long double numerator_sum = 0.0L;
    long double denominator_sum = 0.0L;

    for (std::size_t i = 0; i < actual.size(); ++i)
    {
        const long double diff =
            static_cast<long double>(actual[i]) - static_cast<long double>(reference[i]);
        numerator_sum += diff * diff;

        const long double ref = static_cast<long double>(reference[i]);
        denominator_sum += ref * ref;
    }

    if (denominator_sum == 0.0L)
    {
        return (numerator_sum == 0.0L) ? 0.0 : std::numeric_limits<double>::infinity();
    }

    return static_cast<double>(std::sqrt(numerator_sum / denominator_sum));
}

/**
 * @brief Compute a reference log-determinant using LAPACK.
 *
 * This uses LAPACK's Cholesky factorisation as the reference implementation.
 *
 * @param c Copy of the input matrix.
 * @param n Matrix dimension.
 * @param logdet Output log-determinant on success.
 * @return `true` if LAPACK succeeds.
 */
bool lapack_reference_logdet(std::vector<double> c, int n, LogDetValue& logdet)
{
    const char uplo = 'L';
    const int lda = n;
    int info = 0;

    dpotrf_(&uplo, &n, c.data(), &lda, &info);

    if (info != 0)
    {
        return false;
    }

    logdet = logdet_from_factorised_storage(c, n);
    return true;
}

/**
 * @brief Build the correctness reference data for a benchmark run.
 *
 * @param enabled Whether correctness checking is enabled.
 * @param original Original matrix.
 * @param n Matrix dimension.
 * @return Reference data used for correctness comparison.
 */
CorrectnessReference
make_correctness_reference(bool enabled, const std::vector<double>& original, int n)
{
    CorrectnessReference reference;
    reference.enabled = enabled;

    if (!enabled)
    {
        return reference;
    }

    LogDetValue logdet = 0.0L;
    if (lapack_reference_logdet(original, n, logdet))
    {
        reference.library_available = true;
        reference.logdet_library = logdet;
    }

    return reference;
}

/**
 * @brief Prepare the input matrix and optional correctness reference for a benchmark run.
 *
 * @param n Matrix dimension.
 * @param correctness_enabled Whether correctness checking is enabled.
 * @return Prepared benchmark input data.
 */
BenchmarkInputData prepare_benchmark_input(int n, bool correctness_enabled)
{
    BenchmarkInputData input;
    input.original_matrix = make_generated_spd_matrix(n);
    input.correctness_reference =
        make_correctness_reference(correctness_enabled, input.original_matrix, n);
    return input;
}

/**
 * @brief Evaluate correctness metrics from a factorised result.
 *
 * @param factorised Factorised matrix storage.
 * @param n Matrix dimension.
 * @param reference Stored correctness reference.
 * @return Correctness result for CSV output and reporting.
 */
CorrectnessResult evaluate_correctness_result(const std::vector<double>& factorised,
                                              std::size_t n,
                                              const CorrectnessReference& reference)
{
    CorrectnessResult result;

    if (!reference.enabled)
    {
        return result;
    }

    result.factor_available = true;
    result.logdet_factor = logdet_from_factorised_storage(factorised, n);

    if (reference.library_available)
    {
        result.library_available = true;
        result.relative_difference_available = true;
        result.logdet_library = reference.logdet_library;
        result.relative_difference_percent =
            relative_difference_percent(result.logdet_factor, result.logdet_library);
    }

    return result;
}

/**
 * @brief Write an optional metric to a stream.
 *
 * If the value is unavailable, the text `"unavailable"` is written instead.
 *
 * @param output Output stream.
 * @param available Whether the value is available.
 * @param value Metric value.
 */
void write_optional_metric(std::ostream& output, bool available, LogDetValue value)
{
    if (available)
    {
        output << value;
        return;
    }

    output << "unavailable";
}

/**
 * @brief Format an optional metric for CSV output.
 *
 * @param available Whether the value is available.
 * @param value Metric value.
 * @return CSV-safe string, or an empty field if unavailable.
 */
std::string format_optional_metric_for_csv(bool available, LogDetValue value)
{
    if (!available)
    {
        return "";
    }

    char buffer[128];
    std::snprintf(buffer, sizeof(buffer), "%.21Lg", value);
    return std::string(buffer);
}

/**
 * @brief Format all correctness fields for one CSV row.
 *
 * @param correctness Correctness result.
 * @return Comma-separated correctness fields.
 */
std::string format_correctness_fields_for_csv(const CorrectnessResult& correctness)
{
    std::ostringstream output;
    output << format_optional_metric_for_csv(correctness.library_available,
                                             correctness.logdet_library)
           << ','
           << format_optional_metric_for_csv(correctness.factor_available,
                                             correctness.logdet_factor)
           << ','
           << format_optional_metric_for_csv(correctness.relative_difference_available,
                                             correctness.relative_difference_percent);
    return output.str();
}

/**
 * @brief Overwrite a CSV file with new text.
 *
 * @param path File path.
 * @param text Text to write.
 * @return `true` if the write succeeds.
 */
bool overwrite_csv_text(const std::filesystem::path& path, const std::string& text)
{
    return write_csv_text_with_mode(path, text, "w");
}

/**
 * @brief Append text to a CSV file.
 *
 * @param path File path.
 * @param text Text to append.
 * @return `true` if the write succeeds.
 */
bool append_csv_text(const std::filesystem::path& path, const std::string& text)
{
    return write_csv_text_with_mode(path, text, "a");
}

/**
 * @brief Derive the summary CSV path from the raw CSV path.
 *
 * @param raw_csv_path Raw CSV file path.
 * @return Summary CSV file path.
 */
std::filesystem::path summary_csv_path_for(const std::filesystem::path& raw_csv_path)
{
    const std::string stem = raw_csv_path.stem().string();
    const std::string summary_name = stem.empty() ? "summary.csv" : stem + "_summary.csv";
    return raw_csv_path.parent_path() / summary_name;
}

/**
 * @brief Compute the mean of a list of values.
 *
 * @param values Input values.
 * @return Arithmetic mean, or zero if the input is empty.
 */
double mean_value(const std::vector<double>& values)
{
    if (values.empty())
    {
        return 0.0;
    }

    const double sum = std::accumulate(values.begin(), values.end(), 0.0);
    return sum / static_cast<double>(values.size());
}

/**
 * @brief Compute the median of a list of values.
 *
 * @param values Input values.
 * @return Median, or zero if the input is empty.
 */
double median_value(const std::vector<double>& values)
{
    if (values.empty())
    {
        return 0.0;
    }

    std::vector<double> sorted_values = values;
    std::sort(sorted_values.begin(), sorted_values.end());

    const std::size_t n = sorted_values.size();
    if (n % 2 == 1)
    {
        return sorted_values[n / 2];
    }

    return 0.5 * (sorted_values[n / 2 - 1] + sorted_values[n / 2]);
}

/**
 * @brief Compute the standard deviation of a list of values.
 *
 * @param values Input values.
 * @return Standard deviation, or zero if fewer than two values are present.
 */
double standard_deviation_value(const std::vector<double>& values)
{
    if (values.size() < 2)
    {
        return 0.0;
    }

    const double mu = mean_value(values);
    double sum_sq = 0.0;

    for (const double x : values)
    {
        const double diff = x - mu;
        sum_sq += diff * diff;
    }

    return std::sqrt(sum_sq / static_cast<double>(values.size()));
}

/**
 * @brief Quote a file path for display or shell use.
 *
 * This helps keep paths with spaces readable and safe when printed.
 *
 * @param path File path.
 * @return Quoted path string.
 */
std::string quoted_path(const std::filesystem::path& path)
{
    return "\"" + path.string() + "\"";
}

/**
 * @brief Print the paths of any CSV files written during the run.
 *
 * @param output Output stream.
 * @param csv_session CSV session describing the written files.
 */
void print_successful_csv_writes(std::ostream& output, const BenchmarkCsvSession& csv_session)
{
    output << "Successfully written to file " << quoted_path(csv_session.raw_csv_path()) << '\n';

    if (csv_session.writes_summary())
    {
        output << "Successfully written to file " << quoted_path(csv_session.summary_csv_path())
               << '\n';
    }
}