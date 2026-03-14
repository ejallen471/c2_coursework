/*
perf_scaling.cpp does the following

1. Take one optimisation choice, repeat count and matrix sizes
2. Run the chosen implementation repeatedly for each size
3. Write the raw timing dat to CSV
4. Print a per size summary stats
*/

/**
 * @file perf_scaling.cpp
 * @brief Implementation of the matrix-scaling mode used by run_cholesky.
 */

#include "matrix.h"
#include "perf_helpers.h"
#include "perf_modes.h"
#include "runtime_cholesky.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

namespace
{
// Compute the mean
double mean(const std::vector<double>& values)
{
    if (values.empty())
    {
        return 0.0;
    }

    const double sum = std::accumulate(values.begin(), values.end(), 0.0);
    return sum / static_cast<double>(values.size());
}

// Compute the median
double median(std::vector<double> values)
{
    if (values.empty())
    {
        return 0.0;
    }

    std::sort(values.begin(), values.end());

    // handle even list length case
    const std::size_t n = values.size();
    if (n % 2 == 1)
    {
        return values[n / 2];
    }

    return 0.5 * (values[n / 2 - 1] + values[n / 2]);
}

// Calculate the standard deviation
double standard_deviation(const std::vector<double>& values)
{
    if (values.size() < 2)
    {
        return 0.0;
    }

    const double mu = mean(values);
    double sum_sq = 0.0;

    for (const double x : values)
    {
        const double diff = x - mu;
        sum_sq += diff * diff;
    }

    return std::sqrt(sum_sq / static_cast<double>(values.size()));
}
} // End namespace

int run_scaling_mode(int argc, char* argv[])
{
    if (argc < 5)
    {
        std::cerr << "Usage: " << argv[0]
                  << " <optimisation> <repeats> <raw_csv> <n1> [n2 ...]\n";
        return 1;
    }

    // Create a variable that will store which Cholesky implementation to benchmark
    CholeskyVersion version;
    if (!parse_optimisation_name(argv[1], version))
    {
        std::cerr << "Error: unknown optimisation '" << argv[1] << "'\n";
        return 1;
    }

    const int repeats = std::atoi(argv[2]);
    if (repeats <= 0)
    {
        std::cerr << "Error: repeats must be positive\n";
        return 1;
    }

    // Read the output paths
    const std::filesystem::path raw_csv_path(argv[3]);

    // Create the directories if needed
    if (raw_csv_path.has_parent_path())
    {
        std::filesystem::create_directories(raw_csv_path.parent_path());
    }

    // Open CSV file
    std::ofstream raw_csv(raw_csv_path);
    if (!raw_csv)
    {
        std::cerr << "Error: failed to open raw CSV path: " << raw_csv_path << '\n';
        return 1;
    }

    raw_csv << std::setprecision(kLogDetOutputPrecision); // set the precision for writes to the CSV
    raw_csv << "tag,n,repeat,elapsed_seconds,logdet,time_over_n3\n";

    const std::string tag = optimisation_name(version);
    std::vector<int> matrix_sizes;
    matrix_sizes.reserve(static_cast<std::size_t>(argc - 4));

    for (int argi = 4; argi < argc; ++argi)
    {
        const int n_input = std::atoi(argv[argi]);

        if (n_input <= 0 || n_input > 100000)
        {
            std::cerr << "Error: invalid matrix size '" << argv[argi] << "'\n";
            return 1;
        }

        matrix_sizes.push_back(n_input);
    }

    if (matrix_sizes.empty())
    {
        std::cerr << "Error: at least one matrix size is required\n";
        return 1;
    }

    // Treat each requested matrix size as an independent benchmark group.
    for (const int n_input : matrix_sizes)
    {
        const std::size_t n = static_cast<std::size_t>(n_input); // cast to std::size_t

        const std::vector<double> original = make_generated_spd_matrix(n_input);

        std::vector<double> elapsed_values;
        elapsed_values.reserve(static_cast<std::size_t>(repeats));

        // Time repeated runs from the same original matrix so only the Cholesky version changes.
        for (int repeat = 0; repeat < repeats; ++repeat)
        {
            std::vector<double> working = original;
            const double elapsed = run_cholesky_version(working.data(), n, version);

            if (elapsed < 0.0)
            {
                std::cerr << "Error: factorisation failed for n=" << n << ", repeat=" << repeat
                          << " with code " << elapsed << '\n';
                return 1;
            }

            elapsed_values.push_back(elapsed);

            // Emit enough derived metrics to support later plotting and scaling analysis from CSV
            // alone.
            const LogDetValue logdet = logdet_from_factorised_storage(working, n);
            const double time_over_n3 = elapsed /
                (static_cast<double>(n) * static_cast<double>(n) * static_cast<double>(n));

            raw_csv << tag << ',' << n << ',' << repeat << ',' << elapsed << ',' << logdet << ','
                    << time_over_n3 << '\n';
        }

        // Print a per-size summary to stderr so progress is visible even when stdout is redirected.
        std::cerr << std::setprecision(16);
        std::cerr << "optimisation=" << tag << '\n'
                  << "n=" << n << '\n'
                  << "min_seconds="
                  << *std::min_element(elapsed_values.begin(), elapsed_values.end()) << '\n'
                  << "median_seconds=" << median(elapsed_values) << '\n'
                  << "mean_seconds=" << mean(elapsed_values) << '\n'
                  << "stddev_seconds=" << standard_deviation(elapsed_values) << '\n';
    }

    raw_csv.close();

    std::cout << "raw_csv=" << raw_csv_path << '\n';

    return 0;
}
