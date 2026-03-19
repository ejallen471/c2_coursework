/**
 * @file 03_perf_block_size_sweep.cpp
 * @brief Runs the benchmark mode that measures performance across different block sizes.
 */

#include "matrix.h"
#include "perf_helpers.h"
#include "perf_modes.h"
#include "runtime_cholesky.h"

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace
{
/**
 * @brief Returns the methods that can be used in `block-size-sweep`.
 *
 * This mode is only for implementations that accept an explicit block size.
 * That includes both the single-threaded blocked methods and the blocked OpenMP methods.
 *
 * @return Ordered list of supported blocked implementations.
 */
const std::vector<CholeskyVersion>& block_size_sweep_versions()
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
 * @brief Checks whether a method is valid for `block-size-sweep`.
 *
 * @param version Implementation to check.
 * @return `true` if the method supports an explicit block size in this mode.
 */
bool is_block_size_sweep_version(CholeskyVersion version)
{
    return std::find(block_size_sweep_versions().begin(),
                     block_size_sweep_versions().end(),
                     version) != block_size_sweep_versions().end();
}

/**
 * @brief Parses an optional method name for `block-size-sweep`.
 *
 * This accepts only methods that support an explicit block size.
 *
 * @param input User-supplied method name.
 * @param version Output implementation identifier.
 * @return `true` if the name is valid for this mode.
 */
bool parse_block_size_sweep_version(const std::string& input, CholeskyVersion& version)
{
    return parse_optimisation_name(input, version) && is_block_size_sweep_version(version);
}

/**
 * @brief Builds one raw CSV row.
 *
 * Each row represents a single run for one method, matrix size, block size, and repeat number.
 *
 * @param version The implementation that was run.
 * @param n Matrix size.
 * @param block_size Requested block size.
 * @param repeat Repeat number for this run.
 * @param elapsed Measured runtime in seconds.
 * @param correctness Optional correctness results for this run.
 * @return A CSV row as a string, ending with `\n`.
 */
std::string make_block_size_raw_row(CholeskyVersion version,
                                    std::size_t n,
                                    int block_size,
                                    int repeat,
                                    double elapsed,
                                    const CorrectnessResult& correctness)
{
    std::ostringstream row;
    row << std::setprecision(kLogDetOutputPrecision);
    row << optimisation_name(version) << ',' << n << ',' << block_size << ',' << repeat << ','
        << elapsed << ',' << format_correctness_fields_for_csv(correctness) << '\n';
    return row.str();
}

/**
 * @brief Builds one summary CSV row.
 *
 * The summary row stores the overall timing statistics for one method and one block size.
 *
 * @param version The implementation that was run.
 * @param n Matrix size.
 * @param block_size Requested block size.
 * @param elapsed_values All recorded runtimes for this configuration.
 * @return A CSV row as a string, ending with `\n`.
 */
std::string make_block_size_summary_row(CholeskyVersion version,
                                        std::size_t n,
                                        int block_size,
                                        const std::vector<double>& elapsed_values)
{
    std::ostringstream row;
    row << std::setprecision(kLogDetOutputPrecision);
    row << optimisation_name(version) << ',' << n << ',' << block_size << ','
        << median_value(elapsed_values) << ',' << mean_value(elapsed_values) << ','
        << standard_deviation_value(elapsed_values) << '\n';
    return row.str();
}
} // namespace

/**
 * @brief Runs the `block-size-sweep` benchmark mode.
 *
 * This mode benchmarks one or more blocked Cholesky implementations across a range of
 * block sizes. Each configuration is repeated several times, and the results are written
 * to raw and summary CSV files.
 *
 * An optional thread count can also be provided for OpenMP methods, and correctness
 * checks can be enabled if required.
 *
 * @param argc Number of command-line arguments for this mode.
 * @param argv Argument array for this mode.
 * @return `0` on success, non-zero on failure.
 */
int run_block_size_sweep_mode(int argc, char* argv[])
{
    // Basic argument validation
    if (argc < 5)
    {
        std::cerr << "Usage: " << argv[0]
                  << " <optimisation> <n> <repeats> <raw_csv> "
                     "<block_size1> [block_size2 ...] [--threads N] [--correctness]\n";
        return 1;
    }

    // Parse the optional method name. If none is supplied, all supported blocked methods are used.
    std::vector<CholeskyVersion> selected_versions;
    int numeric_arg_index = 1;

    CholeskyVersion selected_version = CholeskyVersion::Baseline;
    if (parse_block_size_sweep_version(argv[1], selected_version))
    {
        selected_versions.push_back(selected_version);
        numeric_arg_index = 2;
    }
    else
    {
        selected_versions.assign(block_size_sweep_versions().begin(),
                                 block_size_sweep_versions().end());
    }

    // Check that enough arguments remain after parsing the optional method name
    if (argc - numeric_arg_index < 4)
    {
        std::cerr << "Usage: " << argv[0]
                  << " <optimisation> <n> <repeats> <raw_csv> "
                     "<block_size1> [block_size2 ...] [--threads N] [--correctness]\n";
        return 1;
    }

    // Parse matrix size
    int n_input = 0;
    if (!parse_positive_int(argv[numeric_arg_index], n_input) || n_input > 100000)
    {
        std::cerr << "Error: n must be positive and at most 100000\n";
        return 1;
    }
    const std::size_t n = static_cast<std::size_t>(n_input);

    // Parse repeat count
    int repeats = 0;
    if (!parse_positive_int(argv[numeric_arg_index + 1], repeats))
    {
        std::cerr << "Error: repeats must be positive\n";
        return 1;
    }

    const std::filesystem::path raw_csv_path(argv[numeric_arg_index + 2]);

    // Store block sizes, runtime overrides, and correctness flag
    std::vector<int> block_sizes;
    block_sizes.reserve(static_cast<std::size_t>(argc - (numeric_arg_index + 3)));

    int thread_count = 0;
    bool correctness_enabled = false;

    // Parse the remaining arguments
    for (int argi = numeric_arg_index + 3; argi < argc; ++argi)
    {
        const std::string argument = argv[argi];

        if (argument == "--correctness")
        {
            correctness_enabled = true;
            continue;
        }

        if (argument == "--threads")
        {
            if (argi + 1 >= argc || !parse_positive_int(argv[argi + 1], thread_count))
            {
                std::cerr << "Error: --threads requires a positive integer\n";
                return 1;
            }

            ++argi;
            continue;
        }

        int block_size = 0;
        if (!parse_positive_int(argument, block_size))
        {
            std::cerr << "Error: block sizes must be positive integers\n";
            return 1;
        }

        block_sizes.push_back(block_size);
    }

    // At least one block size must be given
    if (block_sizes.empty())
    {
        std::cerr << "Error: at least one block size is required\n";
        return 1;
    }

    for (const CholeskyVersion version : selected_versions)
    {
        if (optimisation_uses_openmp(version) && thread_count <= 0)
        {
            std::cerr << "Error: optimisation '" << optimisation_name(version)
                      << "' requires --threads\n";
            return 1;
        }
    }

    // Prepare the input matrix and optional correctness reference
    const BenchmarkInputData input = prepare_benchmark_input(n_input, correctness_enabled);

    // Initialise the raw and summary CSV files
    const BenchmarkCsvSession csv_session(raw_csv_path, true);
    if (!csv_session.initialise(std::string(kBenchmarkMethodColumnName) +
                                    ",n,block_size,repeat,elapsed_seconds," +
                                    kBenchmarkCorrectnessCsvColumns + "\n",
                                std::cerr,
                                std::string(kBenchmarkMethodColumnName) +
                                    ",n,block_size,elapsed_median,elapsed_mean,elapsed_error\n"))
    {
        return 1;
    }

    // Run every selected method for every requested block size
    for (const CholeskyVersion version : selected_versions)
    {
        for (const int block_size : block_sizes)
        {
            std::vector<double> elapsed_values;

            if (!run_repeated_benchmark_case(
                    input.original_matrix,
                    n,
                    repeats,
                    input.correctness_reference,
                    elapsed_values,

                    // Run one benchmark instance
                    [&](std::vector<double>& working_matrix, int repeat) -> double
                    {
                        const double elapsed = run_cholesky_version_configured(
                            working_matrix.data(),
                            n,
                            version,
                            optimisation_uses_openmp(version) ? thread_count : 0,
                            block_size);

                        if (elapsed < 0.0)
                        {
                            std::cerr << "Error: factorisation failed for "
                                      << optimisation_name(version)
                                      << " with block_size=" << block_size << " repeat=" << repeat
                                      << '\n';
                        }

                        return elapsed;
                    },

                    // Write one raw CSV row
                    [&](int repeat, double elapsed, const CorrectnessResult& correctness) -> bool
                    {
                        return csv_session.append_raw_row(
                            make_block_size_raw_row(
                                version, n, block_size, repeat, elapsed, correctness),
                            std::cerr,
                            std::string(optimisation_name(version)) + " block_size=" +
                                std::to_string(block_size) + " repeat=" + std::to_string(repeat));
                    }))
            {
                return 1;
            }

            // Write one summary row for this method and block size
            if (!csv_session.append_summary_row(
                    make_block_size_summary_row(version, n, block_size, elapsed_values),
                    std::cerr,
                    std::string(optimisation_name(version)) +
                        " block_size=" + std::to_string(block_size)))
            {
                return 1;
            }
        }
    }

    std::cout << "raw_csv=" << csv_session.raw_csv_path() << '\n';
    print_successful_csv_writes(std::cout, csv_session);

    return 0;
}
