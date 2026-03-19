/**
 * @file 04_perf_matrix_size_sweep.cpp
 * @brief Runs the benchmark mode that measures performance across different matrix sizes.
 */

#include "matrix.h"
#include "perf_helpers.h"
#include "perf_modes.h"
#include "runtime_cholesky.h"

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
 * @brief Stores the parsed command-line options for `matrix-size-sweep`.
 *
 * This includes:
 * - the list of matrix sizes to test
 * - any runtime overrides such as thread count or block size
 * - whether correctness checks are enabled
 */
struct MatrixSizeSweepCommandLineOptions
{
    std::vector<int> matrix_sizes;
    int thread_count = 0;
    int block_size = 0;
    bool correctness_enabled = false;
};

/**
 * @brief Builds one raw CSV row.
 *
 * Each row represents a single benchmark run for one matrix size and one repeat.
 *
 * @param method_name User-facing name of the method being tested.
 * @param n Matrix size.
 * @param repeat Repeat number for this run.
 * @param elapsed Measured runtime in seconds.
 * @param correctness Optional correctness results for this run.
 * @return A CSV row as a string, ending with `\n`.
 */
std::string make_matrix_size_raw_row(const std::string& method_name,
                                     std::size_t n,
                                     int repeat,
                                     double elapsed,
                                     const CorrectnessResult& correctness)
{
    std::ostringstream row;
    row << std::setprecision(kLogDetOutputPrecision);
    row << method_name << ',' << n << ',' << repeat << ',' << elapsed << ','
        << format_correctness_fields_for_csv(correctness) << '\n';
    return row.str();
}

/**
 * @brief Builds one summary CSV row.
 *
 * The summary row stores the overall timing statistics for one matrix size.
 *
 * @param method_name User-facing name of the method being tested.
 * @param n Matrix size.
 * @param elapsed_values All recorded runtimes for this matrix size.
 * @return A CSV row as a string, ending with `\n`.
 */
std::string make_matrix_size_summary_row(const std::string& method_name,
                                         std::size_t n,
                                         const std::vector<double>& elapsed_values)
{
    std::ostringstream row;
    row << std::setprecision(kLogDetOutputPrecision);
    row << method_name << ',' << n << ',' << median_value(elapsed_values) << ','
        << mean_value(elapsed_values) << ',' << standard_deviation_value(elapsed_values) << '\n';
    return row.str();
}

/**
 * @brief Parses the command-line options for `matrix-size-sweep`.
 *
 * This reads:
 * - one or more matrix sizes
 * - optional `--threads`
 * - optional `--block-size`
 * - optional `--correctness`
 *
 * @param argc Number of mode-specific arguments.
 * @param argv Array of mode-specific arguments.
 * @param start_index Index of the first argument that may contain sizes or options.
 * @param options Output structure filled on success.
 * @return `true` if all arguments are valid.
 */
bool parse_matrix_size_sweep_options(int argc,
                                     char* argv[],
                                     int start_index,
                                     MatrixSizeSweepCommandLineOptions& options)
{
    for (int argi = start_index; argi < argc; ++argi)
    {
        const std::string argument = argv[argi];

        if (argument == "--correctness")
        {
            options.correctness_enabled = true;
            continue;
        }

        if (argument == "--threads")
        {
            if (argi + 1 >= argc || !parse_positive_int(argv[argi + 1], options.thread_count))
            {
                std::cerr << "Error: --threads requires a positive integer\n";
                return false;
            }

            ++argi;
            continue;
        }

        if (argument == "--block-size")
        {
            if (argi + 1 >= argc || !parse_positive_int(argv[argi + 1], options.block_size))
            {
                std::cerr << "Error: --block-size requires a positive integer\n";
                return false;
            }

            ++argi;
            continue;
        }

        int matrix_size = 0;
        if (!parse_positive_int(argument, matrix_size) || matrix_size > 100000)
        {
            std::cerr << "Error: invalid matrix size '" << argument << "'\n";
            return false;
        }

        options.matrix_sizes.push_back(matrix_size);
    }

    if (options.matrix_sizes.empty())
    {
        std::cerr << "Error: at least one matrix size is required\n";
        return false;
    }

    return true;
}
} // namespace

/**
 * @brief Runs the `matrix-size-sweep` benchmark mode.
 *
 * This mode benchmarks one Cholesky implementation across a range of matrix sizes.
 * Each size is run multiple times, and the results are written to raw and summary CSV files.
 *
 * Optional correctness checks can be enabled, and runtime settings such as thread count
 * and block size can be overridden from the command line.
 *
 * @param argc Number of command-line arguments for this mode.
 * @param argv Argument array for this mode.
 * @return `0` on success, non-zero on failure.
 */
int run_scaling_mode(int argc, char* argv[])
{
    // Basic argument validation
    if (argc < 5)
    {
        std::cerr << "Usage: " << argv[0]
                  << " <optimisation> <repeats> <raw_csv> <n1> [n2 ...] [--threads N] "
                     "[--block-size N] [--correctness]\n";
        return 1;
    }

    // Parse the requested optimisation method
    CholeskyVersion version;
    if (!parse_optimisation_name(argv[1], version))
    {
        std::cerr << "Error: unknown optimisation '" << argv[1] << "'\n";
        return 1;
    }

    // Parse repeat count
    int repeats = 0;
    if (!parse_positive_int(argv[2], repeats))
    {
        std::cerr << "Error: repeats must be positive\n";
        return 1;
    }

    // Open the output CSV session
    const std::filesystem::path raw_csv_path(argv[3]);
    const BenchmarkCsvSession csv_session(raw_csv_path, true);
    if (!csv_session.initialise(std::string(kBenchmarkMethodColumnName) +
                                    ",n,repeat,elapsed_seconds," + kBenchmarkCorrectnessCsvColumns +
                                    "\n",
                                std::cerr,
                                std::string(kBenchmarkMethodColumnName) +
                                    ",n,elapsed_median,elapsed_mean,elapsed_error\n"))
    {
        return 1;
    }

    const std::string method_name = optimisation_name(version);

    // Parse matrix sizes and optional runtime overrides
    MatrixSizeSweepCommandLineOptions options;
    if (!parse_matrix_size_sweep_options(argc, argv, 4, options))
    {
        return 1;
    }

    if (optimisation_uses_openmp(version) && options.thread_count <= 0)
    {
        std::cerr << "Error: optimisation '" << method_name << "' requires --threads\n";
        return 1;
    }

    if (optimisation_supports_block_size(version) && options.block_size <= 0)
    {
        std::cerr << "Error: optimisation '" << method_name << "' requires --block-size\n";
        return 1;
    }

    // Run the benchmark for each requested matrix size
    for (const int n_input : options.matrix_sizes)
    {
        const std::size_t n = static_cast<std::size_t>(n_input);

        // Prepare the input matrix and optional correctness reference
        const BenchmarkInputData input =
            prepare_benchmark_input(n_input, options.correctness_enabled);

        std::vector<double> elapsed_values;

        // Run the selected method repeatedly on this matrix size
        if (!run_repeated_benchmark_case(
                input.original_matrix,
                n,
                repeats,
                input.correctness_reference,
                elapsed_values,

                // Run one timed benchmark instance
                [&](std::vector<double>& working_matrix, int repeat) -> double
                {
                    const double elapsed = run_cholesky_version_configured(
                        working_matrix.data(),
                        n,
                        version,
                        optimisation_uses_openmp(version) ? options.thread_count : 0,
                        optimisation_supports_block_size(version) ? options.block_size : 0);

                    if (elapsed < 0.0)
                    {
                        std::cerr << "Error: factorisation failed for n=" << n
                                  << ", repeat=" << repeat << " with code " << elapsed << '\n';
                    }

                    return elapsed;
                },

                // Write one raw CSV row
                [&](int repeat, double elapsed, const CorrectnessResult& correctness) -> bool
                {
                    return csv_session.append_raw_row(
                        make_matrix_size_raw_row(method_name, n, repeat, elapsed, correctness),
                        std::cerr,
                        "n=" + std::to_string(n) + " repeat=" + std::to_string(repeat));
                }))
        {
            return 1;
        }

        // Write one summary row for this matrix size
        if (!csv_session.append_summary_row(
                make_matrix_size_summary_row(method_name, n, elapsed_values),
                std::cerr,
                "n=" + std::to_string(n)))
        {
            return 1;
        }
    }

    print_successful_csv_writes(std::cout, csv_session);
    return 0;
}
