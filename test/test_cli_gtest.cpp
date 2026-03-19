/**
 * @file test_cli_gtest.cpp
 * @brief GoogleTest coverage for CLI parsing and argument validation.
 */

#include <gtest/gtest.h>

#include "test_suite_helpers.h"

#include <filesystem>
#include <string>
#include <vector>

namespace
{
/**
 * @brief Return whether the supplied text contains a substring.
 *
 * @param text Full text.
 * @param needle Substring to search for.
 * @return `true` when the substring is present.
 */
bool contains_text(const std::string& text, const std::string& needle)
{
    return text.find(needle) != std::string::npos;
}
} // namespace

TEST(CliSuite, RejectsInvalidModesAndMalformedArguments)
{
    ASSERT_TRUE(std::filesystem::exists(benchmark_executable_path()));

    const CommandResult help_result = run_command({benchmark_executable_path().string()});
    EXPECT_NE(help_result.exit_code, 0);
    EXPECT_TRUE(contains_text(help_result.stderr_text, "Usage:"));

    const ScopedTemporaryDirectory temp_dir("cli-invalid");

    const CommandResult unknown_mode = run_benchmark({"unknown-mode"});
    EXPECT_NE(unknown_mode.exit_code, 0);
    EXPECT_TRUE(contains_text(unknown_mode.stderr_text, "unknown mode"));
    EXPECT_TRUE(contains_text(unknown_mode.stderr_text, "Usage:"));

    const CommandResult removed_hyphen_mode = run_benchmark({"single-run"});
    EXPECT_NE(removed_hyphen_mode.exit_code, 0);
    EXPECT_TRUE(contains_text(removed_hyphen_mode.stderr_text, "unknown mode"));

    const CommandResult removed_underscore_mode = run_benchmark({"single_run"});
    EXPECT_NE(removed_underscore_mode.exit_code, 0);
    EXPECT_TRUE(contains_text(removed_underscore_mode.stderr_text, "unknown mode"));

    const CommandResult invalid_method = run_benchmark(
        {"method-compare",
         "4",
         "1",
         (temp_dir.path() / "invalid_method.csv").string(),
         "not_a_method"});
    EXPECT_NE(invalid_method.exit_code, 0);
    EXPECT_TRUE(contains_text(invalid_method.stderr_text, "unknown optimisation"));

    const CommandResult malformed_n = run_benchmark(
        {"method-compare",
         "abc",
         "1",
         (temp_dir.path() / "malformed_n.csv").string(),
         "baseline"});
    EXPECT_NE(malformed_n.exit_code, 0);
    EXPECT_TRUE(contains_text(malformed_n.stderr_text, "n must be between 1 and 100000"));

    const CommandResult malformed_repeats = run_benchmark(
        {"method-compare",
         "4",
         "2abc",
         (temp_dir.path() / "malformed_repeats.csv").string(),
         "baseline"});
    EXPECT_NE(malformed_repeats.exit_code, 0);
    EXPECT_TRUE(contains_text(malformed_repeats.stderr_text, "repeats must be positive"));

    const CommandResult bad_thread_override = run_benchmark(
        {"matrix-size-sweep",
         "openmp_tile_parallel_blocked",
         "1",
         (temp_dir.path() / "bad_thread_override.csv").string(),
         "4",
         "--threads",
         "abc"});
    EXPECT_NE(bad_thread_override.exit_code, 0);
    EXPECT_TRUE(
        contains_text(bad_thread_override.stderr_text, "--threads requires a positive integer"));

    const CommandResult bad_thread_combo = run_benchmark(
        {"thread-count-sweep",
         "8",
         "1",
         (temp_dir.path() / "bad_threads.csv").string(),
         "--threads",
         "1",
         "--methods",
         "baseline"});
    EXPECT_NE(bad_thread_combo.exit_code, 0);
    EXPECT_TRUE(contains_text(
        bad_thread_combo.stderr_text, "not available in thread-count-sweep mode"));
}

TEST(CliSuite, RejectsMalformedOverrideFlags)
{
    const ScopedTemporaryDirectory temp_dir("cli-tuning");

    const CommandResult bad_thread_flag = run_benchmark(
        {"method-compare",
         "8",
         "1",
         (temp_dir.path() / "bad_threads.csv").string(),
         "openmp_tile_parallel_blocked",
         "--threads",
         "abc"});
    EXPECT_NE(bad_thread_flag.exit_code, 0);
    EXPECT_TRUE(contains_text(bad_thread_flag.stderr_text, "invalid --threads"));

    const CommandResult bad_block_size_flag = run_benchmark(
        {"method-compare",
         "8",
         "1",
         (temp_dir.path() / "bad_block_size.csv").string(),
         "cholesky_blocked_tile_kernels",
         "--block-size",
         "abc"});
    EXPECT_NE(bad_block_size_flag.exit_code, 0);
    EXPECT_TRUE(contains_text(bad_block_size_flag.stderr_text, "invalid --block-size"));

    const CommandResult bad_block_override = run_benchmark(
        {"method-compare",
         "8",
         "1",
         (temp_dir.path() / "bad_override.csv").string(),
         "baseline",
         "--block-size-for",
         "baseline"});
    EXPECT_NE(bad_block_override.exit_code, 0);
    EXPECT_TRUE(contains_text(bad_block_override.stderr_text, "invalid block-size override"));
}

TEST(CliSuite, AcceptsAliasesAndValidOverrides)
{
    const ScopedTemporaryDirectory temp_dir("cli-valid");

    const std::filesystem::path alias_csv = temp_dir.path() / "method_alias.csv";
    const CommandResult alias_result = run_benchmark(
        {"method_comparison",
         "8",
         "1",
         alias_csv.string(),
         "choleskyblockedtilekernels",
         "lowertriangle",
         "--block-size",
         "4",
         "--correctness"});
    ASSERT_EQ(alias_result.exit_code, 0) << alias_result.stderr_text;

    const CsvTable alias_table = read_csv_table(alias_csv);
    EXPECT_EQ(alias_table.rows.size(), 2U);

    const std::filesystem::path tuned_csv = temp_dir.path() / "tuned_method_compare.csv";
    const CommandResult tuned_result = run_benchmark(
        {"method-compare",
         "8",
         "1",
         tuned_csv.string(),
         "openmp_tile_parallel_blocked",
         "openmp_tile_list_parallel",
         "--threads",
         "2",
         "--block-size",
         "4",
         "--block-size-for",
         "openmp_tile_list_parallel=6"});
    ASSERT_EQ(tuned_result.exit_code, 0) << tuned_result.stderr_text;

    const CsvTable tuned_table = read_csv_table(tuned_csv);
    EXPECT_EQ(tuned_table.rows.size(), 2U);

    const std::filesystem::path nested_csv =
        temp_dir.path() / "nested" / "path" / "method_compare.csv";
    const CommandResult nested_result = run_benchmark(
        {"method-compare", "4", "1", nested_csv.string(), "baseline", "--correctness"});
    ASSERT_EQ(nested_result.exit_code, 0) << nested_result.stderr_text;
    EXPECT_TRUE(std::filesystem::exists(nested_csv));
}
