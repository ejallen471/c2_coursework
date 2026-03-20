/**
 * @file 01_perf_method_compare.cpp
 * @brief Implements the `method-compare` benchmark mode.
 *
 * This file runs multiple Cholesky implementations on the same matrix size,
 * measures their runtime, and writes results to CSV for comparison.
 */

#include "matrix.h"
#include "perf_helpers.h"
#include "perf_modes.h"
#include "runtime_cholesky.h"

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace // only visible inside this file
{
/**
 * @brief List all implementations available for comparison.
 *
 * This is the full set of methods we can benchmark in this mode.
 */
const std::vector<CholeskyVersion>& method_compare_versions()
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
 * @brief Check if a method is allowed in this mode.
 */
bool is_method_compare_version(CholeskyVersion version)
{
    return std::find(method_compare_versions().begin(), method_compare_versions().end(), version) !=
           method_compare_versions().end();
}

/**
 * @brief Parse method names from the command line.
 *
 * - Converts strings → enum values
 * - Validates them
 * - Removes duplicates
 * - Defaults to all methods if none provided
 */
bool parse_selected_method_compare_versions(int argc,
                                            char* argv[],
                                            int start_index,
                                            std::vector<CholeskyVersion>& versions)
{
    // If no methods provided → use all
    if (argc <= start_index)
    {
        versions.assign(method_compare_versions().begin(), method_compare_versions().end());
        return true;
    }

    // Loop through arguments
    for (int argi = start_index; argi < argc; ++argi)
    {
        const std::string arg = argv[argi];

        // Skip flags (and their values)
        if (arg == "--threads" || arg == "--block-size" || arg == "--block-size-for")
        {
            ++argi;
            continue;
        }

        if (arg == "--correctness")
        {
            continue;
        }

        // Convert string → enum
        CholeskyVersion version = CholeskyVersion::Baseline;
        if (!parse_optimisation_name(arg, version))
        {
            std::cerr << "Error: unknown optimisation '" << arg << "'\n";
            return false;
        }

        // Check allowed
        if (!is_method_compare_version(version))
        {
            std::cerr << "Error: optimisation '" << arg << "' is not available in this mode\n";
            return false;
        }

        // Avoid duplicates
        if (std::find(versions.begin(), versions.end(), version) == versions.end())
        {
            versions.push_back(version);
        }
    }

    return !versions.empty();
}

/**
 * @brief Parse "METHOD=SIZE" block size overrides.
 */
bool parse_block_size_override_spec(const std::string& spec,
                                    CholeskyVersion& version,
                                    int& block_size)
{
    const std::size_t pos = spec.find('=');
    if (pos == std::string::npos || pos == 0 || pos + 1 >= spec.size())
        return false;

    if (!parse_optimisation_name(spec.substr(0, pos), version))
        return false;

    return parse_positive_int(spec.substr(pos + 1), block_size);
}

/**
 * @brief Build one CSV row for a result.
 */
std::string make_method_compare_csv_row(CholeskyVersion version,
                                        std::size_t n,
                                        int repeat,
                                        double elapsed,
                                        bool speedup_available,
                                        double speedup_factor,
                                        const CorrectnessResult& correctness)
{
    std::ostringstream row;
    row << std::setprecision(kLogDetOutputPrecision);

    row << optimisation_name(version) << ',' << n << ',' << repeat << ',' << elapsed << ','
        << format_optional_metric_for_csv(speedup_available, speedup_factor) << ','
        << format_correctness_fields_for_csv(correctness) << '\n';

    return row.str();
}

/**
 * @brief Store the comparison metrics for one matrix generator.
 */
struct MatrixGeneratorComparisonResult
{
    std::string matrix_generator;
    int n = 0;
    double elapsed_seconds = 0.0;
    LogDetValue logdet_library = 0.0L;
    LogDetValue logdet_factor = 0.0L;
    LogDetValue relative_difference_percent = 0.0L;
    double reconstruction_error_frobenius = 0.0;
};

/**
 * @brief Run the baseline factorisation and reference checks for one generated matrix.
 */
bool run_matrix_generator_comparison_case(const std::string& label,
                                          const std::vector<double>& original_matrix,
                                          int n,
                                          MatrixGeneratorComparisonResult& result)
{
    result.matrix_generator = label;
    result.n = n;

    std::vector<double> working_matrix = original_matrix;
    result.elapsed_seconds = timed_cholesky_factorisation(working_matrix.data(), n);
    if (result.elapsed_seconds < 0.0)
    {
        std::cerr << "Error: baseline factorisation failed for " << label
                  << " with code " << result.elapsed_seconds << '\n';
        return false;
    }

    result.logdet_factor =
        logdet_from_factorised_storage(working_matrix, static_cast<std::size_t>(n));
    if (!lapack_reference_logdet(original_matrix, n, result.logdet_library))
    {
        std::cerr << "Error: LAPACK reference factorisation failed for " << label << '\n';
        return false;
    }

    result.relative_difference_percent =
        relative_difference_percent(result.logdet_factor, result.logdet_library);

    const std::vector<double> reconstructed =
        reconstruct_from_factorised_storage(working_matrix, static_cast<std::size_t>(n));
    result.reconstruction_error_frobenius =
        relative_frobenius_error(reconstructed, original_matrix);

    return true;
}

/**
 * @brief Build one CSV row for a matrix-generator comparison result.
 */
std::string make_matrix_generator_compare_csv_row(const MatrixGeneratorComparisonResult& result)
{
    std::ostringstream row;
    row << std::setprecision(kLogDetOutputPrecision);
    row << result.matrix_generator << ','
        << result.n << ','
        << result.elapsed_seconds << ','
        << result.logdet_library << ','
        << result.logdet_factor << ','
        << result.relative_difference_percent << ','
        << result.reconstruction_error_frobenius << '\n';
    return row.str();
}
} // namespace

/**
 * @brief Main implementation of method-compare mode.
 *
 * Steps:
 * 1. Parse inputs (n, repeats, methods, options)
 * 2. Generate matrix once
 * 3. Run each method multiple times
 * 4. Write results immediately to CSV
 */
int run_fixed_size_comparison_mode(int argc, char* argv[])
{
    //  basic argument checking
    if (argc < 4)
    {
        std::cerr << "Usage: " << argv[0]
                  << " <n> <repeats> <raw_csv> [methods...] [--threads N]\n";
        return 1;
    }

    //  parse matrix size
    int n_input = 0;
    if (!parse_positive_int(argv[1], n_input) || n_input > 100000)
    {
        std::cerr << "Error: n must be between 1 and 100000\n";
        return 1;
    }
    const std::size_t n = static_cast<std::size_t>(n_input);

    //  parse repeats
    int repeats = 0;
    if (!parse_positive_int(argv[2], repeats))
    {
        std::cerr << "Error: repeats must be positive\n";
        return 1;
    }

    const std::filesystem::path raw_csv_path(argv[3]);

    //  optional settings
    int thread_count = 0;
    int block_size = 0;
    std::map<CholeskyVersion, int> block_sizes_by_version;
    bool correctness_enabled = false;

    // parse flags
    for (int argi = 4; argi < argc; ++argi)
    {
        const std::string arg = argv[argi];

        if (arg == "--correctness")
        {
            correctness_enabled = true;
        }
        else if (arg == "--threads")
        {
            if (argi + 1 >= argc || !parse_positive_int(argv[argi + 1], thread_count))
            {
                std::cerr << "Error: invalid --threads\n";
                return 1;
            }
            ++argi;
        }
        else if (arg == "--block-size")
        {
            if (argi + 1 >= argc || !parse_positive_int(argv[argi + 1], block_size))
            {
                std::cerr << "Error: invalid --block-size\n";
                return 1;
            }
            ++argi;
        }
        else if (arg == "--block-size-for")
        {
            CholeskyVersion v;
            int bs;
            if (!parse_block_size_override_spec(argv[argi + 1], v, bs))
            {
                std::cerr << "Error: invalid block-size override\n";
                return 1;
            }

            block_sizes_by_version[v] = bs;
            ++argi;
        }
    }

    //  parse selected methods
    std::vector<CholeskyVersion> selected_versions;
    if (!parse_selected_method_compare_versions(argc, argv, 4, selected_versions))
    {
        return 1;
    }

    //  generate matrix once
    const BenchmarkInputData input = prepare_benchmark_input(n_input, correctness_enabled);

    //  setup CSV
    const BenchmarkCsvSession csv_session(raw_csv_path, false);
    if (!csv_session.initialise(std::string(kBenchmarkMethodColumnName) +
                                    ",n,repeat,elapsed_seconds,speedup_factor_vs_baseline," +
                                    kBenchmarkCorrectnessCsvColumns + "\n",
                                std::cerr))
    {
        return 1;
    }

    // check if baseline included
    const bool have_baseline =
        std::find(selected_versions.begin(), selected_versions.end(), CholeskyVersion::Baseline) !=
        selected_versions.end();

    std::vector<double> baseline_times;

    // run baseline first
    if (have_baseline)
    {
        if (!run_repeated_benchmark_case(
            input.original_matrix,
            n,
            repeats,
            input.correctness_reference,
            baseline_times,
            [&](std::vector<double>& mat, int)
            {
                return run_cholesky_version(mat.data(), n, CholeskyVersion::Baseline);
            },
            [&](int r, double t, const CorrectnessResult& c)
            {
                return csv_session.append_raw_row(
                    make_method_compare_csv_row(CholeskyVersion::Baseline, n, r, t, true, 1.0, c),
                    std::cerr,
                    "baseline");
            }))
        {
            return 1;
        }
    }

    //  run other methods
    for (const auto version : selected_versions)
    {
        if (version == CholeskyVersion::Baseline)
            continue;

        std::vector<double> times;

        if (!run_repeated_benchmark_case(
            input.original_matrix,
            n,
            repeats,
            input.correctness_reference,
            times,
            [&](std::vector<double>& mat, int)
            {
                const int effective_block_size =
                    (block_sizes_by_version.count(version) != 0)
                        ? block_sizes_by_version.at(version)
                        : block_size;

                return run_cholesky_version_configured(
                    mat.data(), n, version, thread_count, effective_block_size);
            },
            [&](int r, double t, const CorrectnessResult& c)
            {
                double speedup = have_baseline ? baseline_times[r] / t : 0.0;

                return csv_session.append_raw_row(
                    make_method_compare_csv_row(version, n, r, t, have_baseline, speedup, c),
                    std::cerr,
                    optimisation_name(version));
            }))
        {
            return 1;
        }
    }

    print_successful_csv_writes(std::cout, csv_session);
    return 0;
}

/**
 * @brief Compare the available matrix-generator paths on one matrix size.
 */
int run_matrix_generator_comparison_mode(int argc, char* argv[])
{
    int n_input = 1000;
    if (argc >= 2 && (!parse_positive_int(argv[1], n_input) || n_input > 100000))
    {
        std::cerr << "Error: n must be positive and at most 100000\n";
        return 1;
    }

    if (argc > 3)
    {
        std::cerr << "Usage: " << argv[0] << " [n] [raw_csv]\n";
        return 1;
    }

    const std::filesystem::path raw_csv_path =
        (argc >= 3)
            ? std::filesystem::path(argv[2])
            : std::filesystem::path(
                  "results/raw/matrix_generator_comparison_n" + std::to_string(n_input) + ".csv");

    std::vector<MatrixGeneratorComparisonResult> results;
    results.reserve(2);

    const std::vector<double> coursework_brief_matrix = make_coursework_brief_matrix(n_input);

    MatrixGeneratorComparisonResult coursework_brief_result;
    if (!run_matrix_generator_comparison_case(
            "coursework_brief", coursework_brief_matrix, n_input, coursework_brief_result))
    {
        return 1;
    }
    results.push_back(coursework_brief_result);

    MatrixGeneratorComparisonResult gershgorin_result;
    if (!run_matrix_generator_comparison_case("gershgorin_adjusted_copy",
                                              make_gershgorin_adjusted_copy(
                                                  coursework_brief_matrix, n_input),
                                              n_input,
                                              gershgorin_result))
    {
        return 1;
    }
    results.push_back(gershgorin_result);

    const BenchmarkCsvSession csv_session(raw_csv_path, false);
    if (!csv_session.initialise("matrix_generator,n,elapsed_seconds,logdet_library,"
                                "logdet_factor,relative_difference_percent,"
                                "reconstruction_error_frobenius\n",
                                std::cerr))
    {
        return 1;
    }

    for (const MatrixGeneratorComparisonResult& result : results)
    {
        if (!csv_session.append_raw_row(make_matrix_generator_compare_csv_row(result),
                                        std::cerr,
                                        result.matrix_generator))
        {
            return 1;
        }
    }

    std::cout << std::setprecision(kLogDetOutputPrecision);
    for (const MatrixGeneratorComparisonResult& result : results)
    {
        std::cout << "matrix_generator=" << result.matrix_generator << '\n'
                  << "n=" << result.n << '\n'
                  << "elapsed_seconds=" << result.elapsed_seconds << '\n'
                  << "logdet_library=" << result.logdet_library << '\n'
                  << "logdet_factor=" << result.logdet_factor << '\n'
                  << "relative_difference_percent=" << result.relative_difference_percent << '\n'
                  << "reconstruction_error_frobenius="
                  << result.reconstruction_error_frobenius << '\n';
    }

    print_successful_csv_writes(std::cout, csv_session);
    return 0;
}
