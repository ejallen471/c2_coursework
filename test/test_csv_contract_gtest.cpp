/**
 * @file test_csv_contract_gtest.cpp
 * @brief GoogleTest coverage for benchmark CSV contracts.
 */

#include <gtest/gtest.h>

#include "test_suite_helpers.h"

#include <algorithm>
#include <filesystem>
#include <map>
#include <string>
#include <vector>

namespace
{
/**
 * @brief Return whether every parsed row has an empty value for a field.
 *
 * @param rows Parsed CSV rows.
 * @param field Field name to inspect.
 * @return `true` when all rows have an empty field value.
 */
bool all_rows_have_empty_field(const std::vector<std::map<std::string, std::string>>& rows,
                               const std::string& field)
{
    return std::all_of(
        rows.begin(), rows.end(), [&](const auto& row) { return row.at(field).empty(); });
}

/**
 * @brief Return whether every parsed row has a non-empty value for a field.
 *
 * @param rows Parsed CSV rows.
 * @param field Field name to inspect.
 * @return `true` when all rows have a non-empty field value.
 */
bool all_rows_have_nonempty_field(const std::vector<std::map<std::string, std::string>>& rows,
                                  const std::string& field)
{
    return std::all_of(
        rows.begin(), rows.end(), [&](const auto& row) { return !row.at(field).empty(); });
}

/**
 * @brief Assert that a CSV has the expected header and return the parsed table.
 *
 * @param path CSV file path.
 * @param expected Exact expected header.
 * @return Parsed CSV table.
 */
CsvTable expect_csv_header(const std::filesystem::path& path,
                           const std::vector<std::string>& expected)
{
    const CsvTable table = read_csv_table(path);
    EXPECT_EQ(table.header, expected);
    return table;
}
} // namespace

TEST(CsvContractSuite, MethodCompareContractRemainsStable)
{
    static const std::vector<std::string> kExpectedHeader = {
        "method",
        "n",
        "repeat",
        "elapsed_seconds",
        "speedup_factor_vs_baseline",
        "logdet_library",
        "logdet_factor",
        "relative_difference_percent",
    };

    const ScopedTemporaryDirectory temp_dir("csv-method-compare");

    const std::filesystem::path raw_without_correctness =
        temp_dir.path() / "method_compare_without_correctness.csv";
    const CommandResult without_correctness = run_benchmark(
        {"method-compare",
         "16",
         "2",
         raw_without_correctness.string(),
         "baseline",
         "upper_triangle"});
    ASSERT_EQ(without_correctness.exit_code, 0) << without_correctness.stderr_text;

    const CsvTable without_correctness_table =
        expect_csv_header(raw_without_correctness, kExpectedHeader);
    EXPECT_EQ(without_correctness_table.rows.size(), 4U);
    EXPECT_TRUE(all_rows_have_nonempty_field(
        without_correctness_table.rows, "speedup_factor_vs_baseline"));
    EXPECT_TRUE(all_rows_have_empty_field(without_correctness_table.rows, "logdet_library"));
    EXPECT_TRUE(all_rows_have_empty_field(without_correctness_table.rows, "logdet_factor"));

    const std::filesystem::path raw_with_correctness =
        temp_dir.path() / "method_compare_with_correctness.csv";
    const CommandResult with_correctness = run_benchmark(
        {"method-compare",
         "16",
         "1",
         raw_with_correctness.string(),
         "openmp_tile_parallel_blocked",
         "openmp_tile_list_parallel",
         "--threads",
         "2",
         "--block-size",
         "4",
         "--block-size-for",
         "openmp_tile_list_parallel=6",
         "--correctness"});
    ASSERT_EQ(with_correctness.exit_code, 0) << with_correctness.stderr_text;

    const CsvTable with_correctness_table = expect_csv_header(raw_with_correctness, kExpectedHeader);
    EXPECT_EQ(with_correctness_table.rows.size(), 2U);
    EXPECT_TRUE(all_rows_have_empty_field(
        with_correctness_table.rows, "speedup_factor_vs_baseline"));
    EXPECT_TRUE(all_rows_have_nonempty_field(with_correctness_table.rows, "logdet_library"));
    EXPECT_TRUE(all_rows_have_nonempty_field(with_correctness_table.rows, "logdet_factor"));
    EXPECT_TRUE(all_rows_have_nonempty_field(
        with_correctness_table.rows, "relative_difference_percent"));
}

TEST(CsvContractSuite, MatrixSizeSweepContractRemainsStable)
{
    static const std::vector<std::string> kRawHeader = {
        "method",
        "n",
        "repeat",
        "elapsed_seconds",
        "logdet_library",
        "logdet_factor",
        "relative_difference_percent",
    };
    static const std::vector<std::string> kSummaryHeader = {
        "method",
        "n",
        "elapsed_median",
        "elapsed_mean",
        "elapsed_error",
    };

    const ScopedTemporaryDirectory temp_dir("csv-matrix-size");

    const std::filesystem::path raw_without_correctness =
        temp_dir.path() / "matrix_size_without_correctness.csv";
    const CommandResult without_correctness = run_benchmark(
        {"matrix-size-sweep", "baseline", "2", raw_without_correctness.string(), "4", "8"});
    ASSERT_EQ(without_correctness.exit_code, 0) << without_correctness.stderr_text;

    const CsvTable raw_table = expect_csv_header(raw_without_correctness, kRawHeader);
    EXPECT_EQ(raw_table.rows.size(), 4U);
    EXPECT_TRUE(all_rows_have_empty_field(raw_table.rows, "logdet_factor"));
    EXPECT_TRUE(all_rows_have_empty_field(raw_table.rows, "relative_difference_percent"));

    const CsvTable summary_table =
        expect_csv_header(summary_csv_path(raw_without_correctness), kSummaryHeader);
    EXPECT_EQ(summary_table.rows.size(), 2U);
    EXPECT_EQ(
        std::find(kSummaryHeader.begin(), kSummaryHeader.end(), "time_over_n3"),
        kSummaryHeader.end());

    const std::filesystem::path raw_with_correctness =
        temp_dir.path() / "matrix_size_with_correctness.csv";
    const CommandResult with_correctness = run_benchmark(
        {"matrix-size-sweep",
         "openmp_tile_parallel_blocked",
         "1",
         raw_with_correctness.string(),
         "4",
         "8",
         "--threads",
         "2",
         "--block-size",
         "4",
         "--correctness"});
    ASSERT_EQ(with_correctness.exit_code, 0) << with_correctness.stderr_text;

    const CsvTable with_correctness_table = expect_csv_header(raw_with_correctness, kRawHeader);
    EXPECT_EQ(with_correctness_table.rows.size(), 2U);
    EXPECT_TRUE(all_rows_have_nonempty_field(with_correctness_table.rows, "logdet_library"));
    EXPECT_TRUE(all_rows_have_nonempty_field(with_correctness_table.rows, "logdet_factor"));
    EXPECT_TRUE(all_rows_have_nonempty_field(
        with_correctness_table.rows, "relative_difference_percent"));
}

TEST(CsvContractSuite, BlockSizeSweepContractRemainsStable)
{
    static const std::vector<std::string> kRawHeader = {
        "method",
        "n",
        "block_size",
        "repeat",
        "elapsed_seconds",
        "logdet_library",
        "logdet_factor",
        "relative_difference_percent",
    };
    static const std::vector<std::string> kSummaryHeader = {
        "method",
        "n",
        "block_size",
        "elapsed_median",
        "elapsed_mean",
        "elapsed_error",
    };

    const ScopedTemporaryDirectory temp_dir("csv-block-size");

    const std::filesystem::path raw_without_correctness =
        temp_dir.path() / "block_size_without_correctness.csv";
    const CommandResult without_correctness = run_benchmark(
        {"block-size-sweep",
         "cholesky_blocked_tile_kernels",
         "16",
         "2",
         raw_without_correctness.string(),
         "1",
         "3",
         "8",
         "17"});
    ASSERT_EQ(without_correctness.exit_code, 0) << without_correctness.stderr_text;

    const CsvTable raw_table = expect_csv_header(raw_without_correctness, kRawHeader);
    EXPECT_EQ(raw_table.rows.size(), 8U);
    EXPECT_TRUE(all_rows_have_empty_field(raw_table.rows, "logdet_factor"));
    EXPECT_TRUE(all_rows_have_empty_field(raw_table.rows, "relative_difference_percent"));
    EXPECT_TRUE(std::all_of(
        kRawHeader.begin(),
        kRawHeader.end(),
        [](const std::string& column) { return column.find("speedup") == std::string::npos; }));

    const CsvTable summary_table =
        expect_csv_header(summary_csv_path(raw_without_correctness), kSummaryHeader);
    EXPECT_EQ(summary_table.rows.size(), 4U);

    const std::filesystem::path raw_with_correctness =
        temp_dir.path() / "block_size_with_correctness.csv";
    const CommandResult with_correctness = run_benchmark(
        {"block-size-sweep",
         "cholesky_blocked_tile_kernels",
         "16",
         "1",
         raw_with_correctness.string(),
         "1",
         "3",
         "8",
         "17",
         "--threads",
         "2",
         "--correctness"});
    ASSERT_EQ(with_correctness.exit_code, 0) << with_correctness.stderr_text;

    const CsvTable with_correctness_table = expect_csv_header(raw_with_correctness, kRawHeader);
    EXPECT_EQ(with_correctness_table.rows.size(), 4U);
    EXPECT_TRUE(all_rows_have_nonempty_field(with_correctness_table.rows, "logdet_library"));
    EXPECT_TRUE(all_rows_have_nonempty_field(with_correctness_table.rows, "logdet_factor"));
    EXPECT_TRUE(all_rows_have_nonempty_field(
        with_correctness_table.rows, "relative_difference_percent"));
}

TEST(CsvContractSuite, ThreadCountSweepContractRemainsStable)
{
    static const std::vector<std::string> kRawHeader = {
        "method",
        "n",
        "threads",
        "repeat",
        "elapsed_seconds",
        "logdet_library",
        "logdet_factor",
        "relative_difference_percent",
    };
    static const std::vector<std::string> kSummaryHeader = {
        "method",
        "n",
        "threads",
        "elapsed_median",
        "elapsed_mean",
        "elapsed_error",
    };

    const ScopedTemporaryDirectory temp_dir("csv-thread-count");

    const std::filesystem::path raw_without_correctness =
        temp_dir.path() / "thread_count_without_correctness.csv";
    const CommandResult without_correctness = run_benchmark(
        {"thread-count-sweep",
         "16",
         "2",
         raw_without_correctness.string(),
         "--threads",
         "1",
         "2",
         "--block-size",
         "4",
         "--methods",
         "openmp_row_parallel_unblocked",
         "openmp_tile_parallel_blocked"});
    ASSERT_EQ(without_correctness.exit_code, 0) << without_correctness.stderr_text;

    const CsvTable raw_table = expect_csv_header(raw_without_correctness, kRawHeader);
    EXPECT_EQ(raw_table.rows.size(), 8U);
    EXPECT_TRUE(all_rows_have_empty_field(raw_table.rows, "logdet_factor"));
    EXPECT_TRUE(all_rows_have_empty_field(raw_table.rows, "relative_difference_percent"));

    const CsvTable summary_table =
        expect_csv_header(summary_csv_path(raw_without_correctness), kSummaryHeader);
    EXPECT_EQ(summary_table.rows.size(), 4U);

    const std::filesystem::path raw_with_correctness =
        temp_dir.path() / "thread_count_with_correctness.csv";
    const CommandResult with_correctness = run_benchmark(
        {"thread-count-sweep",
         "16",
         "1",
         raw_with_correctness.string(),
         "--threads",
         "1",
         "2",
         "--block-size",
         "4",
         "--methods",
         "openmp_row_parallel_unblocked",
         "openmp_tile_parallel_blocked",
         "--correctness"});
    ASSERT_EQ(with_correctness.exit_code, 0) << with_correctness.stderr_text;

    const CsvTable with_correctness_table = expect_csv_header(raw_with_correctness, kRawHeader);
    EXPECT_EQ(with_correctness_table.rows.size(), 4U);
    EXPECT_TRUE(all_rows_have_nonempty_field(with_correctness_table.rows, "logdet_library"));
    EXPECT_TRUE(all_rows_have_nonempty_field(with_correctness_table.rows, "logdet_factor"));
    EXPECT_TRUE(all_rows_have_nonempty_field(
        with_correctness_table.rows, "relative_difference_percent"));
}
