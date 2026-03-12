/**
 * @file perf_fixed_size_comparison.cpp
 * @brief Fixed-size benchmark driver that compares all implemented single-threaded methods.
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

struct MethodResult
{
    CholeskyVersion version;
    std::vector<double> elapsed_values;
    std::vector<double> logdet_factor_values;
    std::vector<double> relative_difference_values;
};

const std::vector<CholeskyVersion>& single_thread_versions()
{
    static const std::vector<CholeskyVersion> versions = {
        CholeskyVersion::Baseline,           CholeskyVersion::LowerTriangleOnly,
        CholeskyVersion::InlineMirror,       CholeskyVersion::LoopCleanup,
        CholeskyVersion::AccessPatternAware, CholeskyVersion::CacheBlocked,
        CholeskyVersion::VectorFriendly,     CholeskyVersion::BlockedVectorised,
    };

    return versions;
}
} // namespace

int main(int argc, char* argv[])
{
    if (argc != 4)
    {
        std::cerr << "Usage: " << argv[0] << " <n> <repeats> <raw_csv>\n";
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
    if (raw_csv_path.has_parent_path())
    {
        std::filesystem::create_directories(raw_csv_path.parent_path());
    }

    const std::vector<double> original = make_generated_spd_matrix(n);

    double logdet_library = 0.0;
    if (!lapack_reference_logdet(original, n, logdet_library))
    {
        std::cerr << "Error: LAPACK reference factorisation failed\n";
        return 1;
    }

    std::vector<MethodResult> results;
    results.reserve(single_thread_versions().size());

    for (const CholeskyVersion version : single_thread_versions())
    {
        std::vector<double> warmup = original;
        const double warmup_elapsed = run_cholesky_version(warmup.data(), n, version);
        if (warmup_elapsed < 0.0)
        {
            std::cerr << "Error: warm-up failed for " << optimisation_name(version) << " with code "
                      << warmup_elapsed << '\n';
            return 1;
        }

        MethodResult result;
        result.version = version;
        result.elapsed_values.reserve(static_cast<std::size_t>(repeats));
        result.logdet_factor_values.reserve(static_cast<std::size_t>(repeats));
        result.relative_difference_values.reserve(static_cast<std::size_t>(repeats));

        for (int repeat = 0; repeat < repeats; ++repeat)
        {
            std::vector<double> working = original;
            const double elapsed = run_cholesky_version(working.data(), n, version);
            if (elapsed < 0.0)
            {
                std::cerr << "Error: factorisation failed for " << optimisation_name(version)
                          << " repeat=" << repeat << " with code " << elapsed << '\n';
                return 1;
            }

            const double logdet_factor = logdet_from_factorised_storage(working, n);
            const double relative_difference =
                relative_difference_percent(logdet_factor, logdet_library);

            result.elapsed_values.push_back(elapsed);
            result.logdet_factor_values.push_back(logdet_factor);
            result.relative_difference_values.push_back(relative_difference);
        }

        results.push_back(result);
    }

    const auto baseline_it =
        std::find_if(results.begin(), results.end(), [](const MethodResult& result)
                     { return result.version == CholeskyVersion::Baseline; });
    if (baseline_it == results.end())
    {
        std::cerr << "Error: baseline result missing from fixed-size comparison\n";
        return 1;
    }

    std::ofstream raw_csv(raw_csv_path);
    if (!raw_csv)
    {
        std::cerr << "Error: failed to open raw CSV path: " << raw_csv_path << '\n';
        return 1;
    }

    raw_csv << std::setprecision(16);
    raw_csv << "optimisation,n,repeat,elapsed_seconds,speedup_factor_vs_baseline,"
               "logdet_library,logdet_factor,relative_difference_percent\n";

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
