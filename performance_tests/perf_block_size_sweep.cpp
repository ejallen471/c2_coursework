/**
 * @file perf_block_size_sweep.cpp
 * @brief Implementation of the block-size sweep mode used by run_cholesky.
 */

#include "cholesky_versions.h"
#include "matrix.h"
#include "perf_helpers.h"
#include "perf_modes.h"
#include "runtime_cholesky.h"
#include "timer.h"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace // start the namespace (anonymous space)
{
// Run the baseline - code in src
double run_baseline(double* c, std::size_t n)
{
    const double t0 = wall_time_seconds();
    cholesky_baseline(c, n);
    const double t1 = wall_time_seconds();
    return t1 - t0;
}

// Run the blocked variant (same as above except only the blocked optimisations)
double run_blocked_variant(const std::string& optimisation, double* c, std::size_t n,
                           int block_size)
{
    const double t0 = wall_time_seconds();

    if (optimisation == "cache_blocked_1")
    {
        cholesky_cache_blocked_1(c, n, static_cast<std::size_t>(block_size));
    }
    else if (optimisation == "cache_blocked_2")
    {
        cholesky_cache_blocked_2(c, n, static_cast<std::size_t>(block_size));
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
    // Return the list of blocked Cholesky algorithms names that the should be tested
    static const std::vector<std::string> names = {"cache_blocked_1", "cache_blocked_2"};
    return names;
}

bool parse_blocked_optimisation(const std::string& input, std::string& optimisation)
{
    const std::string name = normalise_optimisation_name(input);

    if (name == "cache_blocked_1" || name == "cacheblocked_1" ||
        name == "cache_blocked1" || name == "cacheblocked1")
    {
        optimisation = "cache_blocked_1";
        return true;
    }

    if (name == "cache_blocked_2" || name == "cacheblocked_2" ||
        name == "cache_blocked2" || name == "cacheblocked2")
    {
        optimisation = "cache_blocked_2";
        return true;
    }

    return false;
}
} // namespace

////////////////////////////// MAIN FUNCTION //////////////////////////////

/*
The main function is the driver for this file it will do

1. read cmd line inputs
2. validate inputs
3. generate test matrix
4. run baseline timings
5. run blocked timings for different block sizes
6. write results to CSV
7. write the CSV summary
*/

int run_block_size_sweep_mode(int argc, char* argv[])
{
    if (argc < 5)
    {
        std::cerr << "Usage: " << argv[0]
                  << " <optimisation> <n> <repeats> <raw_csv> "
                     "<block_size1> [block_size2 ...]\n";
        return 1;
    }

    std::vector<std::string> selected_optimisations;
    int numeric_arg_index = 1;

    std::string selected_optimisation;
    if (parse_blocked_optimisation(argv[1], selected_optimisation))
    {
        selected_optimisations.push_back(selected_optimisation);
        numeric_arg_index = 2;
    }
    else
    {
        selected_optimisations.assign(blocked_optimisations().begin(), blocked_optimisations().end());
    }

    if (argc - numeric_arg_index < 4)
    {
        std::cerr << "Usage: " << argv[0]
                  << " <optimisation> <n> <repeats> <raw_csv> "
                     "<block_size1> [block_size2 ...]\n";
        return 1;
    }

    const int n_input = std::atoi(argv[numeric_arg_index]); // read matrix size

    if (n_input <= 0 || n_input > 100000) // check valid - based on coursework brief
    {
        std::cerr << "Error: n must be positive and at most 100000\n";
        return 1;
    }
    const std::size_t n = static_cast<std::size_t>(n_input); // cast to std::size_t

    const int repeats = std::atoi(argv[numeric_arg_index + 1]); // read repeats and ensure its positive
    if (repeats <= 0)
    {
        std::cerr << "Error: repeats must be positive\n";
        return 1;
    }

    const std::filesystem::path raw_csv_path(argv[numeric_arg_index + 2]); // where to save the CSV

    if (raw_csv_path.has_parent_path()) // check whether the CSV path includes a parent directory
    {
        std::filesystem::create_directories(
            raw_csv_path.parent_path()); // Create the folder if it doesnt already exist
    }

    // Create an empty vector to store all block sizes given on the cmd line
    std::vector<int> block_sizes;
    block_sizes.reserve(static_cast<std::size_t>(argc - (numeric_arg_index + 3))); // reserve enough space in advance

    for (int argi = numeric_arg_index + 3; argi < argc; ++argi) // read block sizes from the cmd line
    {
        const int block_size = std::atoi(argv[argi]);
        if (block_size <= 0)
        {
            std::cerr << "Error: block sizes must be positive integers\n";
            return 1;
        }

        block_sizes.push_back(block_size);
    }

    if (block_sizes.empty())
    {
        std::cerr << "Error: at least one block size is required\n";
        return 1;
    }

    // Create one SPD matrix of size n times n (sotred as const)
    const std::vector<double> original_matrix = make_generated_spd_matrix(n_input);
    std::vector<double> LAPACK_matrix = original_matrix;

    // Compute the LAPACK reference log-determinant
    LogDetValue logdet_library = 0.0L;
    if (!lapack_reference_logdet(LAPACK_matrix, n_input, logdet_library))
    {
        std::cerr << "Error: LAPACK reference factorisation failed\n";
        return 1;
    }

    // Prepare storage for baseline timings
    std::vector<double> baseline_elapsed_values;
    baseline_elapsed_values.reserve(static_cast<std::size_t>(repeats));

    for (int repeat = 0; repeat < repeats; ++repeat) // loop repeats times
    {
        std::vector<double> working_matrix = original_matrix;
        const double elapsed =
            run_baseline(working_matrix.data(), n); // Run the baseline Cholesky implementation
        if (elapsed < 0.0)
        {
            std::cerr << "Error: baseline factorisation failed for repeat=" << repeat << '\n';
            return 1;
        }

        baseline_elapsed_values.push_back(elapsed);
    }

    // Open CSV output file (and check if sucessful)
    std::ofstream raw_csv(raw_csv_path);
    if (!raw_csv)
    {
        std::cerr << "Error: failed to open raw CSV path: " << raw_csv_path << '\n';
        return 1;
    }

    raw_csv << std::setprecision(kLogDetOutputPrecision); // set the output precision

    // Write the CSV header return
    raw_csv << "optimisation,n,block_size,repeat,elapsed_seconds,speedup_factor_vs_baseline,"
               "logdet_library,logdet_factor,relative_difference_percent\n";

    // Loop over optimisation variants
    for (const std::string& optimisation : selected_optimisations)
    {
        // Loop over block sizes
        for (const int block_size : block_sizes)
        {
            // Run the actual runs
            for (int repeat = 0; repeat < repeats; ++repeat)
            {
                std::vector<double> working_matrix = original_matrix;
                const double elapsed =
                    run_blocked_variant(optimisation, working_matrix.data(), n, block_size);
                if (elapsed < 0.0)
                {
                    std::cerr << "Error: factorisation failed for " << optimisation
                              << " with block_size=" << block_size << " repeat=" << repeat << '\n';
                    return 1;
                }

                const LogDetValue logdet_factor = logdet_from_factorised_storage(working_matrix, n);
                const LogDetValue relative_difference =
                    relative_difference_percent(logdet_factor, logdet_library);
                const double speedup_factor = baseline_elapsed_values[repeat] / elapsed;

                // write to CSV
                raw_csv << optimisation << ',' << n << ',' << block_size << ',' << repeat << ','
                        << elapsed << ',' << speedup_factor << ',' << logdet_library << ','
                        << logdet_factor << ',' << relative_difference << '\n';
            }
        }
    }

    raw_csv.close();

    std::cout << "raw_csv=" << raw_csv_path << '\n';

    return 0;
}
