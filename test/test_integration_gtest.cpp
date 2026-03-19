/**
 * @file test_integration_gtest.cpp
 * @brief GoogleTest integration coverage for benchmark modes and incremental writes.
 */

#include <gtest/gtest.h>

#include "test_suite_helpers.h"

#include <chrono>
#include <filesystem>
#include <future>
#include <string>
#include <thread>
#include <vector>

namespace
{
/**
 * @brief Check whether a string contains a substring.
 *
 * @param text Full text.
 * @param needle Substring to find.
 * @return `true` when the substring is present.
 */
bool contains_text(const std::string& text, const std::string& needle)
{
    return text.find(needle) != std::string::npos;
}

/**
 * @brief Assert that a CSV contains a header and at least one data row.
 *
 * @param path CSV file path.
 */
void expect_nonempty_csv(const std::filesystem::path& path)
{
    const CsvTable table = read_csv_table(path);
    EXPECT_FALSE(table.header.empty()) << path;
    EXPECT_FALSE(table.rows.empty()) << path;
}
} // namespace

TEST(IntegrationSuite, MethodCompareOutputAndCsvRemainStable)
{
    ASSERT_TRUE(std::filesystem::exists(benchmark_executable_path()));

    const ScopedTemporaryDirectory temp_dir("integration-method-compare");
    const std::filesystem::path csv_path = temp_dir.path() / "method_compare.csv";

    const CommandResult result = run_benchmark(
        {"method-compare",
         "16",
         "1",
         csv_path.string(),
         "baseline",
         "upper_triangle",
         "--correctness"});
    ASSERT_EQ(result.exit_code, 0) << result.stderr_text;

    EXPECT_TRUE(
        contains_text(result.stdout_text, "Successfully written to file \"" + csv_path.string() + "\""));

    const CsvTable table = read_csv_table(csv_path);
    ASSERT_EQ(
        table.header,
        (std::vector<std::string>{
            "method",
            "n",
            "repeat",
            "elapsed_seconds",
            "speedup_factor_vs_baseline",
            "logdet_library",
            "logdet_factor",
            "relative_difference_percent"}));
    ASSERT_EQ(table.rows.size(), 2U);
    EXPECT_EQ(table.rows[0].at("method"), "baseline");
    EXPECT_EQ(table.rows[0].at("n"), "16");
    EXPECT_EQ(table.rows[0].at("repeat"), "0");
    EXPECT_EQ(table.rows[1].at("method"), "upper_triangle");
    EXPECT_EQ(table.rows[1].at("n"), "16");
    EXPECT_EQ(table.rows[1].at("repeat"), "0");
}

TEST(IntegrationSuite, BenchmarkModesProduceExpectedOutputs)
{
    const ScopedTemporaryDirectory temp_dir("integration-modes");

    const std::filesystem::path method_compare_csv = temp_dir.path() / "method_compare.csv";
    const CommandResult method_compare = run_benchmark(
        {"method-compare",
         "8",
         "1",
         method_compare_csv.string(),
         "openmp_tile_parallel_blocked",
         "openmp_tile_list_parallel",
         "--threads",
         "2",
         "--block-size",
         "4",
         "--block-size-for",
         "openmp_tile_list_parallel=6",
         "--correctness"});
    ASSERT_EQ(method_compare.exit_code, 0) << method_compare.stderr_text;
    expect_nonempty_csv(method_compare_csv);

    const std::filesystem::path matrix_size_csv = temp_dir.path() / "matrix_size.csv";
    const CommandResult matrix_size = run_benchmark(
        {"matrix-size-sweep",
         "openmp_tile_parallel_blocked",
         "1",
         matrix_size_csv.string(),
         "4",
         "8",
         "--threads",
         "2",
         "--block-size",
         "4",
         "--correctness"});
    ASSERT_EQ(matrix_size.exit_code, 0) << matrix_size.stderr_text;
    expect_nonempty_csv(matrix_size_csv);
    expect_nonempty_csv(summary_csv_path(matrix_size_csv));

    const std::filesystem::path block_size_csv = temp_dir.path() / "block_size.csv";
    const CommandResult block_size = run_benchmark(
        {"block-size-sweep",
         "cholesky_blocked_tile_kernels",
         "16",
         "1",
         block_size_csv.string(),
         "1",
         "3",
         "8",
         "--threads",
         "2",
         "--correctness"});
    ASSERT_EQ(block_size.exit_code, 0) << block_size.stderr_text;
    expect_nonempty_csv(block_size_csv);
    expect_nonempty_csv(summary_csv_path(block_size_csv));

    const std::filesystem::path thread_count_csv = temp_dir.path() / "thread_count.csv";
    const CommandResult thread_count = run_benchmark(
        {"thread-count-sweep",
         "16",
         "1",
         thread_count_csv.string(),
         "--threads",
         "1",
         "2",
         "--block-size",
         "4",
         "--methods",
         "openmp_row_parallel_unblocked",
         "openmp_tile_parallel_blocked",
         "--correctness"});
    ASSERT_EQ(thread_count.exit_code, 0) << thread_count.stderr_text;
    expect_nonempty_csv(thread_count_csv);
    expect_nonempty_csv(summary_csv_path(thread_count_csv));
}

TEST(IntegrationSuite, MatrixSizeSweepWritesRowsIncrementally)
{
    using namespace std::chrono_literals;

    const ScopedTemporaryDirectory temp_dir("integration-incremental");
    const std::filesystem::path raw_csv = temp_dir.path() / "incremental_matrix_size.csv";

    const std::vector<std::string> args = {
        "matrix-size-sweep",
        "baseline",
        "40",
        raw_csv.string(),
        "256",
        "288",
        "320",
        "352",
        "384",
        "416",
        "448",
        "480",
        "512",
        "544",
    };

    std::future<CommandResult> future =
        std::async(std::launch::async, [args]() { return run_benchmark(args); });

    bool saw_file = false;
    bool saw_partial_rows_while_running = false;
    const auto deadline = std::chrono::steady_clock::now() + 20s;

    while (std::chrono::steady_clock::now() < deadline)
    {
        const auto status = future.wait_for(0ms);

        if (std::filesystem::exists(raw_csv))
        {
            saw_file = true;
            const std::vector<std::string> lines = read_text_lines(raw_csv);
            if (lines.size() >= 2U && status != std::future_status::ready)
            {
                saw_partial_rows_while_running = true;
                break;
            }
        }

        if (status == std::future_status::ready)
        {
            break;
        }

        std::this_thread::sleep_for(50ms);
    }

    const CommandResult result = future.get();
    EXPECT_EQ(result.exit_code, 0) << result.stderr_text;
    EXPECT_TRUE(saw_file);

    const std::vector<std::string> lines = read_text_lines(raw_csv);
    ASSERT_FALSE(lines.empty());
    EXPECT_EQ(
        lines.front(),
        "method,n,repeat,elapsed_seconds,logdet_library,logdet_factor,relative_difference_percent");
    EXPECT_GE(lines.size(), 2U);

    if (!saw_partial_rows_while_running)
    {
        EXPECT_TRUE(contains_text(
            result.stdout_text, "Successfully written to file \"" + raw_csv.string() + "\""));
    }
}
