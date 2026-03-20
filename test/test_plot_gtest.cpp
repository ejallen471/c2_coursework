/**
 * @file test_plot_gtest.cpp
 * @brief GoogleTest smoke coverage for the plotting CLI.
 */

#include <gtest/gtest.h>

#include "test_suite_helpers.h"

#include <filesystem>
#include <string>

namespace
{
/**
 * @brief Check whether a directory contains a file name with a prefix.
 *
 * @param directory Directory to inspect.
 * @param prefix Filename prefix to search for.
 * @return `true` when any file name starts with the prefix.
 */
bool directory_contains_prefix(const std::filesystem::path& directory, const std::string& prefix)
{
    for (const auto& entry : std::filesystem::directory_iterator(directory))
    {
        if (entry.path().filename().string().rfind(prefix, 0) == 0)
        {
            return true;
        }
    }

    return false;
}

/**
 * @brief Run the plotting CLI and require a successful exit.
 *
 * @param args Plotting command arguments after the Python interpreter path.
 * @return Captured command result.
 */
CommandResult run_plot_command(const std::vector<std::string>& args)
{
    std::vector<std::string> command;
    command.reserve(args.size() + 2U);
    command.push_back(python_executable_path());
    command.push_back(plotter_script_path().string());
    command.insert(command.end(), args.begin(), args.end());

    const CommandResult result = run_command(command, coursework_source_root());
    EXPECT_EQ(result.exit_code, 0) << result.stderr_text;
    return result;
}
} // namespace

TEST(PlotSuite, MatrixSizePlotRunsAndCreatesFigure)
{
    const ScopedTemporaryDirectory temp_dir("plot-matrix-size");
    const std::filesystem::path raw_csv = temp_dir.path() / "matrix_size.csv";
    const std::filesystem::path output_dir = temp_dir.path() / "figures";

    const CommandResult benchmark_result = run_benchmark(
        {"matrix-size-sweep", "baseline", "1", raw_csv.string(), "4", "8", "--correctness"});
    ASSERT_EQ(benchmark_result.exit_code, 0) << benchmark_result.stderr_text;

    const CommandResult plot_result =
        run_plot_command({"matrix-size", raw_csv.string(), output_dir.string()});
    EXPECT_TRUE(std::filesystem::is_directory(output_dir));
    EXPECT_TRUE(std::filesystem::exists(output_dir / "runtime_vs_n.png"));
    EXPECT_NE(plot_result.stdout_text.find("Successful and saved in"), std::string::npos);
}

TEST(PlotSuite, MatrixSizeComparisonPlotRunsAndCreatesFigure)
{
    const ScopedTemporaryDirectory temp_dir("plot-method-comparison");
    const std::filesystem::path baseline_csv = temp_dir.path() / "baseline.csv";
    const std::filesystem::path contiguous_csv = temp_dir.path() / "contiguous.csv";
    const std::filesystem::path output_dir = temp_dir.path() / "figures";

    const CommandResult baseline_result =
        run_benchmark({"matrix-size-sweep", "baseline", "1", baseline_csv.string(), "4", "8"});
    ASSERT_EQ(baseline_result.exit_code, 0) << baseline_result.stderr_text;

    const CommandResult contiguous_result = run_benchmark(
        {"matrix-size-sweep", "contiguous_access", "1", contiguous_csv.string(), "4", "8"});
    ASSERT_EQ(contiguous_result.exit_code, 0) << contiguous_result.stderr_text;

    const CommandResult plot_result = run_plot_command(
        {"matrix-size-comparison",
         output_dir.string(),
         baseline_csv.string(),
         contiguous_csv.string()});
    EXPECT_TRUE(std::filesystem::is_directory(output_dir));
    EXPECT_TRUE(std::filesystem::exists(output_dir / "runtime_vs_n_by_method.png"));
    EXPECT_NE(plot_result.stdout_text.find("Successful and saved in"), std::string::npos);
}

TEST(PlotSuite, BlockSizePlotRunsAndAvoidsDeprecatedSpeedupPlots)
{
    const ScopedTemporaryDirectory temp_dir("plot-block-size");
    const std::filesystem::path raw_csv = temp_dir.path() / "block_size.csv";
    const std::filesystem::path output_dir = temp_dir.path() / "figures";

    const CommandResult benchmark_result = run_benchmark(
        {"block-size-sweep",
         "cholesky_blocked_tile_kernels",
         "16",
         "1",
         raw_csv.string(),
         "1",
         "3",
         "8",
         "--correctness"});
    ASSERT_EQ(benchmark_result.exit_code, 0) << benchmark_result.stderr_text;

    const CommandResult plot_result =
        run_plot_command({"block-size", raw_csv.string(), output_dir.string()});
    EXPECT_TRUE(std::filesystem::is_directory(output_dir));
    EXPECT_TRUE(std::filesystem::exists(output_dir / "runtime_vs_block_size.png"));
    EXPECT_FALSE(directory_contains_prefix(output_dir, "speedup"));
    EXPECT_NE(plot_result.stdout_text.find("Successful and saved in"), std::string::npos);
}

TEST(PlotSuite, ThreadCountPlotRunsAndAvoidsDeprecatedSpeedupPlots)
{
    const ScopedTemporaryDirectory temp_dir("plot-thread-count");
    const std::filesystem::path raw_csv = temp_dir.path() / "thread_count.csv";
    const std::filesystem::path output_dir = temp_dir.path() / "figures";

    const CommandResult benchmark_result = run_benchmark(
        {"thread-count-sweep",
         "16",
         "1",
         raw_csv.string(),
         "--threads",
         "1",
         "2",
         "--block-size",
         "4",
         "--methods",
         "openmp_row_parallel_unblocked",
         "openmp_tile_parallel_blocked",
         "--correctness"});
    ASSERT_EQ(benchmark_result.exit_code, 0) << benchmark_result.stderr_text;

    const CommandResult plot_result =
        run_plot_command({"thread-count", raw_csv.string(), output_dir.string()});
    EXPECT_TRUE(std::filesystem::is_directory(output_dir));
    EXPECT_TRUE(std::filesystem::exists(output_dir / "runtime_vs_thread_count.png"));
    EXPECT_FALSE(directory_contains_prefix(output_dir, "speedup"));
    EXPECT_NE(plot_result.stdout_text.find("Successful and saved in"), std::string::npos);
}

TEST(PlotSuite, ThreadCountPlotInfersOutputDirectoryWhenOmitted)
{
    const ScopedTemporaryDirectory temp_dir("plot-thread-count-auto-output");
    const std::filesystem::path raw_csv = temp_dir.path() / "thread_count.csv";
    const std::filesystem::path output_dir = temp_dir.path() / "figures";

    const CommandResult benchmark_result = run_benchmark(
        {"thread-count-sweep",
         "16",
         "1",
         raw_csv.string(),
         "--threads",
         "1",
         "2",
         "--block-size",
         "4",
         "--methods",
         "openmp_row_parallel_unblocked",
         "openmp_tile_parallel_blocked"});
    ASSERT_EQ(benchmark_result.exit_code, 0) << benchmark_result.stderr_text;

    const CommandResult plot_result = run_plot_command({"thread-count", raw_csv.string()});
    EXPECT_TRUE(std::filesystem::is_directory(output_dir));
    EXPECT_TRUE(std::filesystem::exists(output_dir / "runtime_vs_thread_count.png"));
    EXPECT_NE(plot_result.stdout_text.find("Successful and saved in"), std::string::npos);
}
