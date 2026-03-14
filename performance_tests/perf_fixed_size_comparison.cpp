/**
 * @file perf_fixed_size_comparison.cpp
 * @brief Implementation of the fixed-size comparison mode used by run_cholesky.
 */

#include "matrix.h"
#include "perf_helpers.h"
#include "perf_modes.h"
#include "runtime_cholesky.h"

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace // start namespace (anonymous space)
{
struct MethodResult // Define a data structure used to store the benchmark results for one Cholesky
                    // implementaiton
{
    CholeskyVersion version;            // Store which version
    std::vector<double> elapsed_values; // store runtime measurements
    std::vector<LogDetValue>
        logdet_factor_values; // store the log det computed from the factorised matrix
    std::vector<LogDetValue> relative_difference_values; // store the relative difference (percentage)
};

// Function returns the list of Cholesky implementations (single threaded)
const std::vector<CholeskyVersion>& single_thread_versions()
{
    static const std::vector<CholeskyVersion> versions = {
        CholeskyVersion::Baseline,
        CholeskyVersion::LowerTriangleOnly,
        CholeskyVersion::UpperTriangle,
        CholeskyVersion::ContiguousAccess,
        CholeskyVersion::cacheBlockedOne,
        CholeskyVersion::cacheBlockedTwo,
    };

    return versions;
}
} // End namespace

int run_fixed_size_comparison_mode(int argc, char* argv[])
{
    if (argc < 4) // expects exactly four arguments else error
    {
        std::cerr << "Usage: " << argv[0] << " <n> <repeats> <raw_csv>\n";
        return 1;
    }

    const int n_input = std::atoi(argv[1]);
    if (n_input <= 0 || n_input > 100000)
    {
        std::cerr << "Error: n must be positive and at most 100000\n"; // as per coursework brief
        return 1;
    }
    const std::size_t n = static_cast<std::size_t>(n_input); // cast to std::size_t

    const int repeats = std::atoi(argv[2]);
    if (repeats <= 0)
    {
        std::cerr << "Error: repeats must be positive\n";
        return 1;
    }

    // Read csv output path
    const std::filesystem::path raw_csv_path(argv[3]);
    // ensure the directory exists, otherwise create a folder
    if (raw_csv_path.has_parent_path())
    {
        std::filesystem::create_directories(raw_csv_path.parent_path());
    }

    // generate test matrix  - this matrix is used by all matrices
    const std::vector<double> original_matrix = make_generated_spd_matrix(n_input);
    std::vector<double> LAPACK_matrix = original_matrix;

    // compute results with LAPACK
    LogDetValue logdet_library = 0.0L;
    if (!lapack_reference_logdet(LAPACK_matrix, n_input, logdet_library))
    {
        std::cerr << "Error: LAPACK reference factorisation failed\n";
        return 1;
    }

    // Prepare the storage for results
    std::vector<MethodResult> results;
    results.reserve(single_thread_versions().size());

    // Loop over all Cholesky methods
    for (const CholeskyVersion version : single_thread_versions())
    {
        // Create results container (structure to store the results for this method)
        MethodResult result;
        result.version = version;

        // Reserve the correct amount of storage
        result.elapsed_values.reserve(static_cast<std::size_t>(repeats));
        result.logdet_factor_values.reserve(static_cast<std::size_t>(repeats));
        result.relative_difference_values.reserve(static_cast<std::size_t>(repeats));

        // Loop over the number of repeats
        for (int repeat = 0; repeat < repeats; ++repeat)
        {
            std::vector<double> working_matrix = original_matrix;
            const double elapsed = run_cholesky_version(working_matrix.data(), n, version);
            if (elapsed < 0.0)
            {
                std::cerr << "Error: factorisation failed for " << optimisation_name(version)
                          << " repeat=" << repeat << " with code " << elapsed << '\n';
                return 1;
            }

            const LogDetValue logdet_factor = logdet_from_factorised_storage(working_matrix, n);
            const LogDetValue relative_difference =
                relative_difference_percent(logdet_factor, logdet_library);

            result.elapsed_values.push_back(elapsed);
            result.logdet_factor_values.push_back(logdet_factor);
            result.relative_difference_values.push_back(relative_difference);
        }

        results.push_back(result);
    }

    // Find the baseline method
    const auto baseline_it =
        std::find_if(results.begin(), results.end(), [](const MethodResult& result)
                     { return result.version == CholeskyVersion::Baseline; });
    if (baseline_it == results.end())
    {
        std::cerr << "Error: baseline result missing from fixed-size comparison\n";
        return 1;
    }

    // Open CSV file
    std::ofstream raw_csv(raw_csv_path);
    if (!raw_csv)
    {
        std::cerr << "Error: failed to open raw CSV path: " << raw_csv_path << '\n';
        return 1;
    }

    // Write the CSV header
    raw_csv << std::setprecision(kLogDetOutputPrecision);
    raw_csv << "optimisation,n,repeat,elapsed_seconds,speedup_factor_vs_baseline,"
               "logdet_library,logdet_factor,relative_difference_percent\n";

    // Write the results
    for (const MethodResult& result : results)
    {
        for (std::size_t repeat = 0; repeat < result.elapsed_values.size(); ++repeat)
        {
            const double elapsed = result.elapsed_values[repeat];
            const double speedup_factor = baseline_it->elapsed_values[repeat] / elapsed;

            raw_csv << optimisation_name(result.version) << ',' << n << ',' << repeat << ','
                    << elapsed << ',' << speedup_factor << ',' << logdet_library << ','
                    << result.logdet_factor_values[repeat] << ','
                    << result.relative_difference_values[repeat] << '\n';
        }
    }

    raw_csv.close();

    std::cout << "raw_csv=" << raw_csv_path << '\n';

    return 0;
}
