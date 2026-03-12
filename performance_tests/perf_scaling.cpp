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
/**
 * @brief Recover the log-determinant from an in-place Cholesky factorisation.
 *
 * The factorised storage contains the lower-triangular factor `L` such that `A = L L^T`.
 *
 * @param c Dense row-major matrix storage containing the Cholesky factor.
 * @param n Matrix dimension.
 * @return The value of `log(det(A))`.
 */
double logdet_from_factorised_storage(const std::vector<double>& c, int n)
{
    double sum = 0.0;

    for (int i = 0; i < n; ++i)
    {
        const std::size_t index =
            static_cast<std::size_t>(i) * static_cast<std::size_t>(n) + static_cast<std::size_t>(i);
        sum += std::log(c[index]);
    }

    // Cholesky stores A = L L^T, so log(det(A)) = 2 * sum(log(diag(L))).
    return 2.0 * sum;
}

/**
 * @brief Compute the dense storage requirement for an `n x n` matrix in bytes.
 *
 * @param n Matrix dimension.
 * @return Number of bytes required for dense double-precision storage.
 */
double matrix_bytes(int n)
{
    return static_cast<double>(n) * static_cast<double>(n) * sizeof(double);
}

/**
 * @brief Estimate the floating-point work of dense Cholesky factorisation.
 *
 * @param n Matrix dimension.
 * @return Approximate flop count using the standard `n^3 / 3` estimate.
 */
double cholesky_flop_estimate(int n)
{
    // Dense Cholesky is approximately n^3 / 3 floating-point operations.
    return (1.0 / 3.0) * static_cast<double>(n) * static_cast<double>(n) * static_cast<double>(n);
}

/**
 * @brief Compute the arithmetic mean of a list of values.
 *
 * @param values Input data.
 * @return Mean of the values, or `0.0` if the input is empty.
 */
double mean(const std::vector<double>& values)
{
    if (values.empty())
    {
        return 0.0;
    }

    const double sum = std::accumulate(values.begin(), values.end(), 0.0);
    return sum / static_cast<double>(values.size());
}

/**
 * @brief Compute the median of a list of values.
 *
 * @param values Input data copied by value so it can be sorted locally.
 * @return Median of the values, or `0.0` if the input is empty.
 */
double median(std::vector<double> values)
{
    if (values.empty())
    {
        return 0.0;
    }

    std::sort(values.begin(), values.end());

    const std::size_t n = values.size();
    if (n % 2 == 1)
    {
        return values[n / 2];
    }

    return 0.5 * (values[n / 2 - 1] + values[n / 2]);
}

/**
 * @brief Compute the population standard deviation of a list of values.
 *
 * @param values Input data.
 * @return Standard deviation, or `0.0` if fewer than two values are provided.
 */
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

/**
 * @brief Wrap a filesystem path in quotes for safe use in a shell command string.
 *
 * @param path Filesystem path to quote.
 * @return Quoted string representation of the path.
 */
std::string quoted_path(const std::filesystem::path& path)
{
    // Quote paths before passing them to std::system so spaces do not break the command line.
    return "\"" + path.string() + "\"";
}
} // namespace

/**
 * @brief Run repeated timings for several matrix sizes and generate scaling outputs.
 *
 * Expected usage:
 * `perf_scaling <optimisation> <repeats> <raw_csv> <plot_output_dir> <n1> [n2 ...]`
 *
 * The program writes raw benchmark rows to the requested CSV, prints per-size summaries to `stderr`,
 * and finally invokes the Python plotting script to produce processed summaries and figures.
 *
 * @param argc Number of command-line arguments.
 * @param argv Command-line argument array containing the optimisation name, repeat count, output paths,
 * and one or more matrix sizes.
 * @return `0` on success, or a non-zero value if argument parsing, benchmarking, file output, or plotting fails.
 */
int main(int argc, char* argv[])
{
    if (argc < 6)
    {
        std::cerr << "Usage: " << argv[0]
                  << " <optimisation> <repeats> <raw_csv> <plot_output_dir> <n1> [n2 ...]\n";
        return 1;
    }

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

    const std::filesystem::path raw_csv_path(argv[3]);
    const std::filesystem::path plot_output_dir(argv[4]);

    if (raw_csv_path.has_parent_path())
    {
        std::filesystem::create_directories(raw_csv_path.parent_path());
    }
    std::filesystem::create_directories(plot_output_dir);

    std::ofstream raw_csv(raw_csv_path);
    if (!raw_csv)
    {
        std::cerr << "Error: failed to open raw CSV path: " << raw_csv_path << '\n';
        return 1;
    }

    raw_csv << std::setprecision(16);
    raw_csv << "tag,n,repeat,elapsed_seconds,logdet,matrix_bytes,flop_estimate,gflops_est,"
               "time_over_n3\n";

    const std::string tag = optimisation_name(version);

    // Treat each requested matrix size as an independent benchmark group.
    for (int argi = 5; argi < argc; ++argi)
    {
        const int n = std::atoi(argv[argi]);
        if (n <= 0 || n > 100000)
        {
            std::cerr << "Error: invalid matrix size '" << argv[argi] << "'\n";
            return 1;
        }

        const std::vector<double> original = make_generated_spd_matrix(n);
        const double bytes = matrix_bytes(n);
        const double flop_est = cholesky_flop_estimate(n);

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
            const double logdet = logdet_from_factorised_storage(working, n);
            const double gflops_est = flop_est / elapsed / 1.0e9;
            const double time_over_n3 = elapsed /
                (static_cast<double>(n) * static_cast<double>(n) * static_cast<double>(n));

            raw_csv << tag << ',' << n << ',' << repeat << ',' << elapsed << ',' << logdet << ','
                    << bytes << ',' << flop_est << ',' << gflops_est << ',' << time_over_n3 << '\n';
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
