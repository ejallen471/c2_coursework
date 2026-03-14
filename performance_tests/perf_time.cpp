/*
perf_time.cpp

This file does the following

1. Read in an optimsation name and matrix size from the cmd line
2. Generate one SPD matrix of that size (using matrix.h and matrix.cpp)
3. Run the chosen cholesky implementation with timing (all through choleksy.cpp)
4. Print the time taken

*/

/**
 * @file perf_time.cpp
 * @brief Implementation of the single-run benchmark mode used by run_cholesky.
 */

#include "matrix.h"
#include "perf_helpers.h"
#include "perf_modes.h"
#include "runtime_cholesky.h"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace // start namespace
{
} // End namespace

int run_time_mode(int argc, char* argv[])
{
    if (argc != 3 && argc != 4)
    {
        std::cerr << "Usage: " << argv[0] << " <optimisation> <n> [raw_csv]\n";
        return 1;
    }

    CholeskyVersion version;
    if (!parse_optimisation_name(argv[1], version))
    {
        std::cerr << "Error: unknown optimisation '" << argv[1] << "'\n";
        return 1;
    }

    const int n_input = std::atoi(argv[2]);
    if (n_input <= 0 || n_input > 100000)
    {
        std::cerr << "Error: n must be positive and at most 100000\n";
        return 1;
    }
    const std::size_t n = static_cast<std::size_t>(n_input); // cast to std::size_t
    const bool write_csv = argc == 4;
    const std::filesystem::path raw_csv_path = write_csv ? std::filesystem::path(argv[3])
                                                         : std::filesystem::path();

    if (write_csv && raw_csv_path.has_parent_path())
    {
        std::filesystem::create_directories(raw_csv_path.parent_path());
    }

    // Generate one dense SPD test matrix, keep one copy for the reference library calculation,
    const std::vector<double> original = make_generated_spd_matrix(n_input);
    std::vector<double> c = original;

    // run the optimisation variant implementation
    const double elapsed = run_cholesky_version(c.data(), n, version);

    // error checking
    if (elapsed < 0.0)
    {
        std::cerr << "Error: factorisation failed with code " << elapsed << '\n';
        return 1;
    }

    // Calculate the log-determinant from the returned factor
    const LogDetValue computed_logdet = logdet_from_factorised_storage(c, n);

    // Calculate the log-determinant from LAPACK when available.
    LogDetValue library_logdet = 0.0L;
    const bool have_library_reference = lapack_reference_logdet(original, n_input, library_logdet);

    std::cout << std::setprecision(kLogDetOutputPrecision);
    std::cout << "optimisation=" << optimisation_name(version) << '\n';
    std::cout << "n=" << n << '\n';
    std::cout << "elapsed_seconds=" << elapsed << '\n';

    if (have_library_reference)
    {
        const LogDetValue relative_diff_pct =
            relative_difference_percent(computed_logdet, library_logdet);
        std::cout << "logdet_library=" << library_logdet << '\n';
        std::cout << "logdet_factor=" << computed_logdet << '\n';
        std::cout << "relative_difference_percent=" << relative_diff_pct << '\n';
    }
    else
    {
        std::cout << "logdet_library=unavailable\n";
        std::cout << "logdet_factor=" << computed_logdet << '\n';
        std::cout << "relative_difference_percent=unavailable\n";
    }

    if (write_csv)
    {
        std::ofstream raw_csv(raw_csv_path);
        if (!raw_csv)
        {
            std::cerr << "Error: failed to open raw CSV path: " << raw_csv_path << '\n';
            return 1;
        }

        raw_csv << std::setprecision(kLogDetOutputPrecision);
        raw_csv << "optimisation,n,elapsed_seconds,logdet_library,logdet_factor,"
                   "relative_difference_percent\n";

        raw_csv << optimisation_name(version) << ',' << n << ',' << elapsed << ',';
        if (have_library_reference)
        {
            raw_csv << library_logdet << ',' << computed_logdet << ','
                    << relative_difference_percent(computed_logdet, library_logdet) << '\n';
        }
        else
        {
            raw_csv << "unavailable," << computed_logdet << ",unavailable\n";
        }

        raw_csv.close();
        std::cout << "raw_csv=" << raw_csv_path << '\n';
    }

    return 0;
}
