/**
 * @file perf_helpers.h
 * @brief Shared helper functions and data structures used by the benchmark modes.
 */

#ifndef PERF_HELPERS_H
#define PERF_HELPERS_H

#include <cstddef>
#include <filesystem>
#include <limits>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

/// Numeric type used for log-determinant calculations and CSV output.
using LogDetValue = long double;

/// Precision used when writing `LogDetValue` results.
inline constexpr int kLogDetOutputPrecision = std::numeric_limits<LogDetValue>::max_digits10;

/// Standard CSV column name used for method labels across benchmark modes.
inline constexpr const char* kBenchmarkMethodColumnName = "method";

/// Standard correctness columns used by all raw benchmark CSV files.
inline constexpr const char* kBenchmarkCorrectnessCsvColumns =
    "logdet_library,logdet_factor,relative_difference_percent";

/**
 * @brief Stores the reusable correctness reference for one benchmark input.
 *
 * When correctness checking is enabled, this holds the LAPACK-based reference value
 * that can be reused across repeated runs on the same input matrix.
 */
struct CorrectnessReference
{
    bool enabled = false;              ///< Whether correctness checking was requested.
    bool library_available = false;    ///< Whether the LAPACK reference was computed successfully.
    LogDetValue logdet_library = 0.0L; ///< Reference `log(det(A))` computed by LAPACK.
};

/**
 * @brief Stores correctness values for one benchmark run.
 *
 * This holds the log-determinant reconstructed from the factorised result, together with
 * the optional LAPACK reference value and the percentage difference between them.
 */
struct CorrectnessResult
{
    bool factor_available = false;  ///< Whether the factor-based log-determinant was computed.
    bool library_available = false; ///< Whether the LAPACK reference value is available.
    bool relative_difference_available =
        false;                         ///< Whether the percentage-difference field is valid.
    LogDetValue logdet_library = 0.0L; ///< Reference `log(det(A))` from LAPACK.
    LogDetValue logdet_factor = 0.0L;  ///< `log(det(A))` reconstructed from the factorised matrix.
    LogDetValue relative_difference_percent =
        0.0L; ///< Percentage difference from the reference value.
};

/**
 * @brief Stores the generated input matrix and its optional correctness reference.
 */
struct BenchmarkInputData
{
    std::vector<double> original_matrix;        ///< Source matrix reused across repeated runs.
    CorrectnessReference correctness_reference; ///< Optional reusable correctness reference.
};

/**
 * @brief Manages the raw and summary CSV files for one benchmark run.
 *
 * This helper owns the output paths, creates parent directories, writes the CSV headers,
 * and appends complete rows while keeping the current flush-after-each-row behaviour.
 */
class BenchmarkCsvSession
{
public:
    /**
     * @brief Create a CSV session for one raw CSV path.
     *
     * @param raw_csv_path Path to the raw CSV file.
     * @param writes_summary Whether this mode also writes a summary CSV file.
     */
    BenchmarkCsvSession(std::filesystem::path raw_csv_path, bool writes_summary);

    /**
     * @brief Create the output files and write their header rows.
     *
     * @param raw_header Header line for the raw CSV, ending in `\n`.
     * @param error_output Stream used for reporting errors.
     * @param summary_header Header line for the summary CSV when one is needed.
     * @return `true` if all requested files were created successfully.
     */
    bool initialise(const std::string& raw_header,
                    std::ostream& error_output,
                    const std::string& summary_header = "") const;

    /**
     * @brief Append one complete row to the raw CSV file.
     *
     * @param row Fully formatted CSV row ending in `\n`.
     * @param error_output Stream used for reporting errors.
     * @param context Optional extra context for any error message.
     * @return `true` if the row was written successfully.
     */
    bool append_raw_row(const std::string& row,
                        std::ostream& error_output,
                        const std::string& context = "") const;

    /**
     * @brief Append one complete row to the summary CSV file.
     *
     * @param row Fully formatted CSV row ending in `\n`.
     * @param error_output Stream used for reporting errors.
     * @param context Optional extra context for any error message.
     * @return `true` if the row was written successfully.
     */
    bool append_summary_row(const std::string& row,
                            std::ostream& error_output,
                            const std::string& context = "") const;

    /**
     * @brief Return the raw CSV path for this session.
     */
    const std::filesystem::path& raw_csv_path() const;

    /**
     * @brief Return the summary CSV path for this session.
     *
     * @note This is only meaningful when `writes_summary()` is `true`.
     */
    const std::filesystem::path& summary_csv_path() const;

    /**
     * @brief Report whether this session writes a summary CSV file.
     */
    bool writes_summary() const;

private:
    std::filesystem::path raw_csv_path_;
    std::filesystem::path summary_csv_path_;
    bool writes_summary_ = false;
};

/**
 * @brief Parse a string as an integer and reject partial or invalid input.
 *
 * @param text Input text.
 * @param value Output integer value on success.
 * @return `true` if the full string is a valid decimal integer.
 */
bool parse_strict_int(const std::string& text, int& value);

/**
 * @brief Parse a string as a positive integer and reject partial or invalid input.
 *
 * @param text Input text.
 * @param value Output integer value on success.
 * @return `true` if the full string is a valid positive integer.
 */
bool parse_positive_int(const std::string& text, int& value);

/**
 * @brief Compute `log(det(A))` from a factorised matrix.
 *
 * The matrix is assumed to contain a Cholesky factor in mirrored storage.
 *
 * @param c Factorised matrix storage.
 * @param n Matrix dimension.
 * @return Log-determinant reconstructed from the diagonal of the factor.
 */
LogDetValue logdet_from_factorised_storage(const std::vector<double>& c, std::size_t n);

/**
 * @brief Compute the relative percentage difference between two values.
 *
 * @param value Measured value.
 * @param reference Reference value.
 * @return Percentage difference relative to the magnitude of `reference`.
 */
LogDetValue relative_difference_percent(LogDetValue value, LogDetValue reference);

/**
 * @brief Reconstruct the original matrix from mirrored Cholesky storage.
 *
 * @param c Factorised matrix storage.
 * @param n Matrix dimension.
 * @return Reconstructed dense row-major matrix.
 */
std::vector<double> reconstruct_from_factorised_storage(const std::vector<double>& c,
                                                        std::size_t n);

/**
 * @brief Compute the relative Frobenius norm error between two matrices.
 *
 * @param actual Measured matrix values.
 * @param reference Reference matrix values.
 * @return Relative Frobenius error, or infinity if the shapes do not match.
 */
double relative_frobenius_error(const std::vector<double>& actual,
                                const std::vector<double>& reference);

/**
 * @brief Compute a reference log-determinant using LAPACK.
 *
 * @param c Copy of the source matrix. LAPACK factorises it in place.
 * @param n Matrix dimension.
 * @param logdet Output log-determinant on success.
 * @return `true` if the LAPACK factorisation succeeds.
 */
bool lapack_reference_logdet(std::vector<double> c, int n, LogDetValue& logdet);

/**
 * @brief Build the reusable correctness reference for one source matrix.
 *
 * @param enabled Whether correctness checking is enabled.
 * @param original Source matrix.
 * @param n Matrix dimension.
 * @return Reference data for later per-run correctness checks.
 */
CorrectnessReference
make_correctness_reference(bool enabled, const std::vector<double>& original, int n);

/**
 * @brief Generate one benchmark input matrix and its optional correctness reference.
 *
 * @param n Matrix dimension.
 * @param correctness_enabled Whether correctness checking is enabled.
 * @return Source matrix plus any reusable correctness reference.
 */
BenchmarkInputData prepare_benchmark_input(int n, bool correctness_enabled);

/**
 * @brief Compute the correctness values for one factorised result.
 *
 * @param factorised Factorised matrix storage for one benchmark run.
 * @param n Matrix dimension.
 * @param reference Precomputed correctness reference.
 * @return Per-run correctness result, including availability flags.
 */
CorrectnessResult evaluate_correctness_result(const std::vector<double>& factorised,
                                              std::size_t n,
                                              const CorrectnessReference& reference);

/**
 * @brief Write either a numeric value or `unavailable` to a stream.
 *
 * @param output Target stream.
 * @param available Whether the numeric value is valid.
 * @param value Value to write when it is available.
 */
void write_optional_metric(std::ostream& output, bool available, LogDetValue value);

/**
 * @brief Format one optional metric for CSV output.
 *
 * @param available Whether the numeric value is valid.
 * @param value Value to serialise when available.
 * @return Numeric text, or an empty string if unavailable.
 */
std::string format_optional_metric_for_csv(bool available, LogDetValue value);

/**
 * @brief Format the standard correctness fields for one raw benchmark CSV row.
 *
 * @param correctness Per-run correctness result.
 * @return CSV text for `logdet_library`, `logdet_factor`, and `relative_difference_percent`.
 */
std::string format_correctness_fields_for_csv(const CorrectnessResult& correctness);

/**
 * @brief Replace a CSV file with new text.
 *
 * @param path Destination CSV path.
 * @param text Full text to write, usually a header line.
 * @return `true` on success.
 */
bool overwrite_csv_text(const std::filesystem::path& path, const std::string& text);

/**
 * @brief Append text to a CSV file.
 *
 * @param path Destination CSV path.
 * @param text Full text to append, usually a complete row.
 * @return `true` on success.
 */
bool append_csv_text(const std::filesystem::path& path, const std::string& text);

/**
 * @brief Return the summary CSV path that matches a raw benchmark CSV path.
 *
 * @param raw_csv_path Raw CSV path.
 * @return Summary CSV path using the `_summary.csv` suffix.
 */
std::filesystem::path summary_csv_path_for(const std::filesystem::path& raw_csv_path);

/**
 * @brief Compute the arithmetic mean of a set of values.
 *
 * @param values Input values.
 * @return Mean, or `0` if the input is empty.
 */
double mean_value(const std::vector<double>& values);

/**
 * @brief Compute the median of a set of values.
 *
 * @param values Input values.
 * @return Median, or `0` if the input is empty.
 */
double median_value(const std::vector<double>& values);

/**
 * @brief Compute the population standard deviation of a set of values.
 *
 * @param values Input values.
 * @return Standard deviation, or `0` if fewer than two values are present.
 */
double standard_deviation_value(const std::vector<double>& values);

/**
 * @brief Wrap a filesystem path in double quotes for safe display.
 *
 * @param path Path to quote.
 * @return Quoted string form of the path.
 */
std::string quoted_path(const std::filesystem::path& path);

/**
 * @brief Print success messages for the CSV files written by one benchmark run.
 *
 * @param output Stream used for the user-facing success messages.
 * @param csv_session Benchmark CSV session whose files were written successfully.
 */
void print_successful_csv_writes(std::ostream& output, const BenchmarkCsvSession& csv_session);

/**
 * @brief Run a repeated benchmark case using the shared correctness workflow.
 *
 * Each repeat starts from a fresh copy of the original matrix, computes correctness
 * from the finished factor, and passes the completed row data to `consume_row`
 * straight away so callers can append and flush raw CSV output one row at a time.
 *
 * @tparam MeasureRun Callable with signature `double(std::vector<double>&, int)`.
 * @tparam ConsumeRow Callable with signature
 *     `bool(int repeat, double elapsed, const CorrectnessResult& correctness)`.
 * @param original_matrix Source matrix reused across repeats.
 * @param n Matrix dimension.
 * @param repeats Number of repeats to run.
 * @param correctness_reference Optional reusable correctness reference.
 * @param elapsed_values Output vector filled with elapsed times.
 * @param measure_run Callable that runs one factorisation and returns elapsed seconds.
 * @param consume_row Callable that consumes one completed repeat result.
 * @return `true` if every repeat completed successfully.
 *
 * @note `measure_run` should return a negative value after reporting its own error message.
 */
template <typename MeasureRun, typename ConsumeRow>
bool run_repeated_benchmark_case(const std::vector<double>& original_matrix,
                                 std::size_t n,
                                 int repeats,
                                 const CorrectnessReference& correctness_reference,
                                 std::vector<double>& elapsed_values,
                                 MeasureRun&& measure_run,
                                 ConsumeRow&& consume_row)
{
    elapsed_values.clear();
    elapsed_values.reserve(static_cast<std::size_t>(repeats));

    for (int repeat = 0; repeat < repeats; ++repeat)
    {
        std::vector<double> working_matrix = original_matrix;
        const double elapsed = measure_run(working_matrix, repeat);
        if (elapsed < 0.0)
        {
            return false;
        }

        elapsed_values.push_back(elapsed);

        const CorrectnessResult correctness =
            evaluate_correctness_result(working_matrix, n, correctness_reference);
        if (!consume_row(repeat, elapsed, correctness))
        {
            return false;
        }
    }

    return true;
}

#endif