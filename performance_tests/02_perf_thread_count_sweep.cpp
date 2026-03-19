/**
 * @file 02_perf_thread_count_sweep.cpp
 * @brief Runs the benchmark mode that measures performance across different OpenMP thread counts.
 */

#include "matrix.h"
#include "perf_helpers.h"
#include "perf_modes.h"
#include "runtime_cholesky.h"

#include <algorithm>
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
 * @brief Returns the methods that can be used in `thread-count-sweep`.
 *
 * This mode is only intended for OpenMP implementations. Including single-threaded
 * methods would be misleading, as changing the thread count would have no effect.
 *
 * @return Ordered list of supported OpenMP implementations.
 */
const std::vector<CholeskyVersion>& thread_count_sweep_versions()
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
 * @brief Checks whether a method is valid for this mode.
 *
 * @param version Implementation to check.
 * @return `true` if the method is supported in `thread-count-sweep`.
 */
bool is_thread_count_sweep_version(CholeskyVersion version)
{
    return std::find(thread_count_sweep_versions().begin(),
                     thread_count_sweep_versions().end(),
                     version) != thread_count_sweep_versions().end();
}

/**
 * @brief Builds one raw CSV row.
 *
 * Each row represents a single run with a given method, matrix size,
 * thread count, and repeat index.
 */
std::string make_thread_count_raw_row(CholeskyVersion version,
                                      std::size_t n,
                                      int thread_count,
                                      int repeat,
                                      double elapsed,
                                      const CorrectnessResult& correctness)
{
    std::ostringstream row;
    row << std::setprecision(kLogDetOutputPrecision);

    row << optimisation_name(version) << ',' << n << ',' << thread_count << ',' << repeat << ','
        << elapsed << ',' << format_correctness_fields_for_csv(correctness) << '\n';

    return row.str();
}

/**
 * @brief Builds one summary CSV row.
 *
 * The summary aggregates all repeats for a given method and thread count.
 */
std::string make_thread_count_summary_row(CholeskyVersion version,
                                          std::size_t n,
                                          int thread_count,
                                          const std::vector<double>& elapsed_values)
{
    std::ostringstream row;
    row << std::setprecision(kLogDetOutputPrecision);

    row << optimisation_name(version) << ',' << n << ',' << thread_count << ','
        << median_value(elapsed_values) << ',' << mean_value(elapsed_values) << ','
        << standard_deviation_value(elapsed_values) << '\n';

    return row.str();
}
} // namespace

/**
 * @brief Runs the `thread-count-sweep` benchmark mode.
 *
 * This mode runs one or more OpenMP implementations across a range of thread counts.
 * Each configuration is repeated multiple times, and results are written to CSV files.
 */
int run_thread_count_sweep_mode(int argc, char* argv[])
{
    // Basic argument validation
    if (argc < 6)
    {
        std::cerr << "Usage: " << argv[0]
                  << " <n> <repeats> <raw_csv> --threads <t1> [t2 ...] "
                     "[--methods <m1> [m2 ...]] [--block-size N] [--correctness]\n";
        return 1;
    }

    // Parse matrix size
    int n_input = 0;
    if (!parse_positive_int(argv[1], n_input) || n_input > 100000)
    {
        std::cerr << "Error: n must be positive and at most 100000\n";
        return 1;
    }
    const std::size_t n = static_cast<std::size_t>(n_input);

    // Parse repeat count
    int repeats = 0;
    if (!parse_positive_int(argv[2], repeats))
    {
        std::cerr << "Error: repeats must be positive\n";
        return 1;
    }

    const std::filesystem::path raw_csv_path(argv[3]);

    // Runtime configuration flags
    bool correctness_enabled = false;
    int block_size_override = 0;

    std::vector<int> thread_counts;
    std::vector<CholeskyVersion> selected_versions;

    // Track which section we are parsing
    enum class ParseSection
    {
        None,
        Threads,
        Methods,
    };

    ParseSection section = ParseSection::None;

    // Parse remaining arguments
    for (int argi = 4; argi < argc; ++argi)
    {
        const std::string argument = argv[argi];

        if (argument == "--correctness")
        {
            correctness_enabled = true;
            continue;
        }

        if (argument == "--threads")
        {
            section = ParseSection::Threads;
            continue;
        }

        if (argument == "--methods")
        {
            section = ParseSection::Methods;
            continue;
        }

        if (argument == "--block-size")
        {
            if (argi + 1 >= argc || !parse_positive_int(argv[argi + 1], block_size_override))
            {
                std::cerr << "Error: --block-size requires a positive integer\n";
                return 1;
            }

            ++argi;
            continue;
        }

        // Parse thread counts
        if (section == ParseSection::Threads)
        {
            int thread_count = 0;
            if (!parse_positive_int(argv[argi], thread_count))
            {
                std::cerr << "Error: thread counts must be positive integers\n";
                return 1;
            }

            thread_counts.push_back(thread_count);
            continue;
        }

        // Parse method names
        if (section == ParseSection::Methods)
        {
            CholeskyVersion version = CholeskyVersion::Baseline;

            if (!parse_optimisation_name(argument, version))
            {
                std::cerr << "Error: unknown optimisation '" << argument << "'\n";
                return 1;
            }

            if (!is_thread_count_sweep_version(version))
            {
                std::cerr << "Error: optimisation '" << argument
                          << "' is not available in thread-count-sweep mode\n";
                return 1;
            }

            // Avoid duplicates
            if (std::find(selected_versions.begin(), selected_versions.end(), version) ==
                selected_versions.end())
            {
                selected_versions.push_back(version);
            }

            continue;
        }

        // Unknown argument
        std::cerr << "Error: unexpected argument '" << argument
                  << "'. Use --threads and optional --methods.\n";
        return 1;
    }

    // Ensure at least one thread count was provided
    if (thread_counts.empty())
    {
        std::cerr << "Error: at least one thread count must be provided after --threads\n";
        return 1;
    }

    // Default to all methods if none specified
    if (selected_versions.empty())
    {
        selected_versions.assign(thread_count_sweep_versions().begin(),
                                 thread_count_sweep_versions().end());
    }

    for (const CholeskyVersion version : selected_versions)
    {
        if (optimisation_supports_block_size(version) && block_size_override <= 0)
        {
            std::cerr << "Error: optimisation '" << optimisation_name(version)
                      << "' requires --block-size\n";
            return 1;
        }
    }

    // Prepare input matrix (and optional correctness reference)
    const BenchmarkInputData input = prepare_benchmark_input(n_input, correctness_enabled);

    // Initialise CSV output
    const BenchmarkCsvSession csv_session(raw_csv_path, true);
    if (!csv_session.initialise(std::string(kBenchmarkMethodColumnName) +
                                    ",n,threads,repeat,elapsed_seconds," +
                                    kBenchmarkCorrectnessCsvColumns + "\n",
                                std::cerr,
                                std::string(kBenchmarkMethodColumnName) +
                                    ",n,threads,elapsed_median,elapsed_mean,elapsed_error\n"))
    {
        return 1;
    }

    // Main benchmark loop
    for (const CholeskyVersion version : selected_versions)
    {
        for (const int thread_count : thread_counts)
        {
            std::vector<double> elapsed_values;

            // Run repeated measurements
            if (!run_repeated_benchmark_case(
                    input.original_matrix,
                    n,
                    repeats,
                    input.correctness_reference,
                    elapsed_values,

                    // Run one instance
                    [&](std::vector<double>& working_matrix, int repeat) -> double
                    {
                        const double elapsed = run_cholesky_version_configured(
                            working_matrix.data(),
                            n,
                            version,
                            thread_count,
                            optimisation_supports_block_size(version) ? block_size_override : 0);

                        if (elapsed < 0.0)
                        {
                            std::cerr << "Error: factorisation failed for "
                                      << optimisation_name(version) << " threads=" << thread_count
                                      << " repeat=" << repeat << " with code " << elapsed << '\n';
                        }

                        return elapsed;
                    },

                    // Write raw CSV row
                    [&](int repeat, double elapsed, const CorrectnessResult& correctness) -> bool
                    {
                        return csv_session.append_raw_row(
                            make_thread_count_raw_row(
                                version, n, thread_count, repeat, elapsed, correctness),
                            std::cerr,
                            std::string(optimisation_name(version)) + " threads=" +
                                std::to_string(thread_count) + " repeat=" + std::to_string(repeat));
                    }))
            {
                return 1;
            }

            // Write summary row
            if (!csv_session.append_summary_row(
                    make_thread_count_summary_row(version, n, thread_count, elapsed_values),
                    std::cerr,
                    std::string(optimisation_name(version)) +
                        " threads=" + std::to_string(thread_count)))
            {
                return 1;
            }
        }
    }

    print_successful_csv_writes(std::cout, csv_session);
    return 0;
}
