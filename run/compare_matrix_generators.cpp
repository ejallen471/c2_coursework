/**
 * @file compare_matrix_generators.cpp
 * @brief Compares the coursework matrix generators against baseline factorisation metrics.
 */

#include "matrix.h"
#include "mphil_dis_cholesky.h"
#include "perf_helpers.h"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace
{
struct ComparisonResult
{
    std::string matrix_generator;
    int n = 0;
    double elapsed_seconds = 0.0;
    LogDetValue logdet_library = 0.0L;
    LogDetValue logdet_factor = 0.0L;
    LogDetValue relative_difference_percent = 0.0L;
    double reconstruction_error_frobenius = 0.0;
};

bool run_one(const std::string& label,
             const std::vector<double>& original_matrix,
             int n,
             ComparisonResult& result)
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

    result.logdet_factor = logdet_from_factorised_storage(working_matrix, static_cast<std::size_t>(n));
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
} // namespace

int main(int argc, char* argv[])
{
    const int n = (argc >= 2) ? std::atoi(argv[1]) : 1000;
    if (n <= 0 || n > 100000)
    {
        std::cerr << "Error: n must be positive and at most 100000\n";
        return 1;
    }

    std::filesystem::path raw_csv_path =
        (argc >= 3) ? std::filesystem::path(argv[2])
                    : std::filesystem::path("results/raw/matrix_generator_comparison_n" +
                                            std::to_string(n) + ".csv");

    if (raw_csv_path.has_parent_path())
    {
        std::filesystem::create_directories(raw_csv_path.parent_path());
    }

    std::vector<ComparisonResult> results;
    results.reserve(2);

    const std::vector<double> coursework_brief_matrix = make_coursework_brief_matrix(n);

    ComparisonResult coursework_brief_result;
    if (!run_one("coursework_brief", coursework_brief_matrix, n, coursework_brief_result))
    {
        return 1;
    }
    results.push_back(coursework_brief_result);

    ComparisonResult gershgorin_result;
    if (!run_one("gershgorin_adjusted_copy",
                 make_gershgorin_adjusted_copy(coursework_brief_matrix, n),
                 n,
                 gershgorin_result))
    {
        return 1;
    }
    results.push_back(gershgorin_result);

    std::ofstream raw_csv(raw_csv_path);
    if (!raw_csv)
    {
        std::cerr << "Error: failed to open raw CSV path: " << raw_csv_path << '\n';
        return 1;
    }

    raw_csv << std::setprecision(kLogDetOutputPrecision);
    raw_csv << "matrix_generator,n,elapsed_seconds,logdet_library,logdet_factor,"
               "relative_difference_percent,reconstruction_error_frobenius\n";

    for (const ComparisonResult& result : results)
    {
        raw_csv << result.matrix_generator << ','
                << result.n << ','
                << result.elapsed_seconds << ','
                << result.logdet_library << ','
                << result.logdet_factor << ','
                << result.relative_difference_percent << ','
                << result.reconstruction_error_frobenius << '\n';
    }

    std::cout << std::setprecision(kLogDetOutputPrecision);
    for (const ComparisonResult& result : results)
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

    std::cout << "raw_csv=" << raw_csv_path << '\n';
    return 0;
}
