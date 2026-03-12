/**
 * @file perf_block_size_sweep.cpp
 * @brief Benchmark blocked Cholesky variants across a user-provided list of block sizes.
 */

#include "cholesky_versions.h"
#include "matrix.h"
#include "timer.h"

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

extern "C"
{
void dpotrf_(const char* uplo, const int* n, double* a, const int* lda, int* info);
}

namespace
{
double logdet_from_factorised_storage(const std::vector<double>& c, int n)
{
    double sum = 0.0;

    for (int i = 0; i < n; ++i)
    {
        const std::size_t index =
            static_cast<std::size_t>(i) * static_cast<std::size_t>(n) + static_cast<std::size_t>(i);
        sum += std::log(c[index]);
    }

    return 2.0 * sum;
}

double relative_difference_percent(double value, double reference)
{
    const double scale = std::fabs(reference);
    if (scale == 0.0)
    {
        return (std::fabs(value) == 0.0) ? 0.0 : 100.0;
    }

    return 100.0 * std::fabs(value - reference) / scale;
}

bool lapack_reference_logdet(std::vector<double> c, int n, double& logdet)
{
    const char uplo = 'L';
    const int lda = n;
    int info = 0;

    dpotrf_(&uplo, &n, c.data(), &lda, &info);
    if (info != 0)
    {
        return false;
    }

    logdet = logdet_from_factorised_storage(c, n);
    return true;
}

double run_baseline(double* c, int n)
{
    const double t0 = wall_time_seconds();
    cholesky_baseline(c, static_cast<std::size_t>(n));
    const double t1 = wall_time_seconds();
    return t1 - t0;
}

double run_blocked_variant(const std::string& optimisation, double* c, int n, int block_size)
{
    const double t0 = wall_time_seconds();

    if (optimisation == "cache_blocked")
    {
        cholesky_cache_blocked(c, static_cast<std::size_t>(n), block_size);
    }
    else if (optimisation == "blocked_vectorised")
    {
        cholesky_blocked_vectorised(c, static_cast<std::size_t>(n), block_size);
    }
    else
    {
        return -1.0;
    }

    const double t1 = wall_time_seconds();
    return t1 - t0;
}

const std::vector<std::string>& blocked_optimisations()
{
    static const std::vector<std::string> names = {"cache_blocked", "blocked_vectorised"};
    return names;
}
} // namespace

int main(int argc, char* argv[])
{
    if (argc < 6)
    {
        std::cerr << "Usage: " << argv[0]
                  << " <n> <repeats> <raw_csv> <plot_output_dir> <block_size1> [block_size2 ...]\n";
        return 1;
    }

    const int n = std::atoi(argv[1]);
    if (n <= 0 || n > 100000)
    {
        std::cerr << "Error: n must be positive and at most 100000\n";
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

    std::vector<int> block_sizes;
    block_sizes.reserve(static_cast<std::size_t>(argc - 5));

    for (int argi = 5; argi < argc; ++argi)
    {
        const int block_size = std::atoi(argv[argi]);
        if (block_size <= 0)
        {
            std::cerr << "Error: block sizes must be positive integers\n";
            return 1;
        }

        block_sizes.push_back(block_size);
    }

    const std::vector<double> original = make_generated_spd_matrix(n);

    double logdet_library = 0.0;
    if (!lapack_reference_logdet(original, n, logdet_library))
    {
        std::cerr << "Error: LAPACK reference factorisation failed\n";
        return 1;
    }

    std::vector<double> baseline_elapsed_values;
    baseline_elapsed_values.reserve(static_cast<std::size_t>(repeats));

    {
        std::vector<double> warmup = original;
        const double warmup_elapsed = run_baseline(warmup.data(), n);
        if (warmup_elapsed < 0.0)
        {
            std::cerr << "Error: baseline warm-up failed\n";
            return 1;
        }
    }

    for (int repeat = 0; repeat < repeats; ++repeat)
    {
        std::vector<double> working = original;
        const double elapsed = run_baseline(working.data(), n);
        if (elapsed < 0.0)
        {
            std::cerr << "Error: baseline factorisation failed for repeat=" << repeat << '\n';
            return 1;
        }

        baseline_elapsed_values.push_back(elapsed);
    }

    std::ofstream raw_csv(raw_csv_path);
    if (!raw_csv)
    {
        std::cerr << "Error: failed to open raw CSV path: " << raw_csv_path << '\n';
        return 1;
    }

    raw_csv << std::setprecision(16);
    raw_csv << "optimisation,n,block_size,repeat,elapsed_seconds,speedup_factor_vs_baseline,"
               "logdet_library,logdet_factor,relative_difference_percent\n";

    for (const std::string& optimisation : blocked_optimisations())
    {
        for (const int block_size : block_sizes)
        {
            {
                std::vector<double> warmup = original;
                const double warmup_elapsed =
                    run_blocked_variant(optimisation, warmup.data(), n, block_size);
                if (warmup_elapsed < 0.0)
                {
                    std::cerr << "Error: warm-up failed for " << optimisation
                              << " with block_size=" << block_size << '\n';
                    return 1;
                }
            }

            for (int repeat = 0; repeat < repeats; ++repeat)
            {
                std::vector<double> working = original;
                const double elapsed =
                    run_blocked_variant(optimisation, working.data(), n, block_size);
                if (elapsed < 0.0)
                {
                    std::cerr << "Error: factorisation failed for " << optimisation
                              << " with block_size=" << block_size
                              << " repeat=" << repeat << '\n';
                    return 1;
                }

                const double logdet_factor = logdet_from_factorised_storage(working, n);
                const double relative_difference =
                    relative_difference_percent(logdet_factor, logdet_library);
                const double speedup_factor = baseline_elapsed_values[repeat] / elapsed;

                raw_csv << optimisation << ',' << n << ',' << block_size << ',' << repeat << ','
                        << elapsed << ',' << speedup_factor << ',' << logdet_library << ','
                        << logdet_factor << ',' << relative_difference << '\n';
            }
        }
    }

    raw_csv.close();

    const std::filesystem::path plot_script_path =
        std::filesystem::path(MPHIL_PROJECT_SOURCE_DIR) / "plot" / "plot_block_size_metrics.py";
    const std::string plot_command = "python3 \"" + plot_script_path.string() + "\" \"" +
                                     raw_csv_path.string() + "\" \"" + plot_output_dir.string() +
                                     "\"";

    const int plot_status = std::system(plot_command.c_str());
    if (plot_status != 0)
    {
        std::cerr << "Error: plotting command failed with status " << plot_status << '\n';
        return 1;
    }

    std::cout << "raw_csv=" << raw_csv_path << '\n';
    std::cout << "plot_output_dir=" << plot_output_dir << '\n';

    return 0;
}
