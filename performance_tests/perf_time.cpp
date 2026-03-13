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
 * @brief Single-run benchmark driver for timing one Cholesky implementation on one generated
 * matrix.
 */

#include "matrix.h"
#include "perf_helpers.h"
#include "runtime_cholesky.h"

#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace // start namespace
{
} // End namespace

//////////////////////// MAIN FUNCTION ////////////////////////

int main(int argc, char* argv[])
{
    if (argc != 3)
    {
        std::cerr << "Usage: " << argv[0] << " <optimisation> <n>\n";
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

    // Calculate the log-determinant from LAPACK
    LogDetValue library_logdet = 0.0L;
    if (!lapack_reference_logdet(original, n_input, library_logdet))
    {
        std::cerr << "Error: LAPACK reference factorisation failed\n";
        return 1;
    }

    // Calculate the relative error
    const LogDetValue relative_diff_pct = relative_difference_percent(computed_logdet, library_logdet);

    std::cout << std::setprecision(kLogDetOutputPrecision);
    std::cout << "optimisation=" << optimisation_name(version) << " n=" << n
              << " elapsed_seconds=" << elapsed;

    std::cout << " logdet_library=" << library_logdet << " logdet_factor=" << computed_logdet
              << " relative_difference_percent=" << relative_diff_pct;

    std::cout << '\n';

    return 0;
}
