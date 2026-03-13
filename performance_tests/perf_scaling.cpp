/*
perf_scaling.cpp does the following

1. Take one optimisation choice, repeat count and matrix sizes
2. Run the chosen implementation repeatedly for each size
3. Write the raw timing dat to CSV
6. Print a per size summary stats
7. Call the python plotting script to generate graphs
*/

/**
 * @file perf_scaling.cpp
 * @brief Multi-size benchmark driver that records repeated timings and generates scaling plots.
 */

#include "matrix.h"
#include "perf_helpers.h"
#include "runtime_cholesky.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
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

//////////////////////// MAIN FUNCTION ////////////////////////

/*
The main does the following

1. reads command-line inputs
2. chooses which Cholesky version to benchmark
3. opens the CSV output file
4. loops over matrix sizes
5. warms up once for each size
6. runs timed repeats for each size
7. writes timing data and derived metrics to CSV
8. prints summary statistics
9. runs a Python plotting script
10. prints where the outputs were saved
*/

int main(int argc, char* argv[])
{
    if (argc < 6) // check argument count - expect at least six
    {
        std::cerr << "Usage: " << argv[0]
                  << " <optimisation> <repeats> <raw_csv> <plot_output_dir> [--warmup|--no-warmup] <n1> [n2 ...]\n";
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
    const std::filesystem::path plot_output_dir(argv[4]);

    // Create the directories if needed
    if (raw_csv_path.has_parent_path())
    {
        std::filesystem::create_directories(raw_csv_path.parent_path());
    }
    std::filesystem::create_directories(plot_output_dir);

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
    bool run_warmup = true;
    std::vector<int> matrix_sizes;
    matrix_sizes.reserve(static_cast<std::size_t>(argc - 5));

    for (int argi = 5; argi < argc; ++argi)
    {
        if (parse_warmup_option(argv[argi], run_warmup))
        {
            continue;
        }

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

        if (run_warmup)
        {
            // Do one untimed warm-up on a copy so first-run effects do not pollute the reported
            // repeats.
            {
                std::vector<double> warmup = original;
                const double elapsed = run_cholesky_version(warmup.data(), n, version);

                if (elapsed < 0.0)
                {
                    std::cerr << "Error: warm-up factorisation failed for n=" << n << " with code "
                              << elapsed << '\n';
                    return 1;
                }
            }
        }

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
        std::cerr << "optimisation=" << tag << " n=" << n << " min_seconds="
                  << *std::min_element(elapsed_values.begin(), elapsed_values.end())
                  << " median_seconds=" << median(elapsed_values)
                  << " mean_seconds=" << mean(elapsed_values)
                  << " stddev_seconds=" << standard_deviation(elapsed_values) << '\n';
    }

    raw_csv.close();

    const std::filesystem::path plot_script =
        std::filesystem::path(MPHIL_PROJECT_SOURCE_DIR) / "plot" / "plot_metrics.py";
    // Delegate graph generation to the Python plotting script after all raw timing data is written.
    const std::string command = "python3 " + quoted_path(plot_script) + " " +
        quoted_path(raw_csv_path) + " " + quoted_path(plot_output_dir);

    const int plot_status = std::system(command.c_str());
    if (plot_status != 0)
    {
        std::cerr << "Error: plotting command failed: " << command << '\n';
        return 1;
    }

    std::cout << "raw_csv=" << raw_csv_path << '\n';
    std::cout << "plot_output_dir=" << plot_output_dir << '\n';

    return 0;
}
