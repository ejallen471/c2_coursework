/**
 * @file perf_block_size_sweep.cpp
 * @brief Benchmark blocked Cholesky variants across a user-provided list of block sizes.
 */

#include "cholesky_guard.h"
#include "cholesky_versions.h"
#include "matrix.h"
#include "perf_helpers.h"
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
    const double guard = cholesky_detail::factorised_matrix_guard(c, n);
    cholesky_detail::consume_cholesky_guard(guard); // keep the factorisation observable
    return t1 - t0;
}

// Run the blocked variant (same as above except only the blocked optimisations)
double run_blocked_variant(const std::string& optimisation, double* c, std::size_t n,
                           int block_size)
{
    const double t0 = wall_time_seconds();

    if (optimisation == "cache_blocked")
    {
        cholesky_cache_blocked(c, n, block_size);
    }
    else if (optimisation == "blocked_vectorised")
    {
        cholesky_blocked_vectorised(c, n, block_size);
    }
    else
    {
        return -1.0;
    }

    const double t1 = wall_time_seconds();
    const double guard = cholesky_detail::factorised_matrix_guard(c, n);
    cholesky_detail::consume_cholesky_guard(guard);
    return t1 - t0;
}

const std::vector<std::string>& blocked_optimisations()
{
    // Return the list of blocked Cholesky algorithms names that the should be tested
    static const std::vector<std::string> names = {"cache_blocked", "blocked_vectorised"};
    return names;
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
7. call python plotting script
*/

int main(int argc, char* argv[])
{
    if (argc < 6) // must take 7 or more inputs, if less error and tell user
    {
        std::cerr << "Usage: " << argv[0]
                  << " <n> <repeats> <raw_csv> <plot_output_dir> [--warmup|--no-warmup] "
                     "<block_size1> [block_size2 ...]\n";
        return 1;
    }

    const int n_input = std::atoi(argv[1]); // read matrix size

    if (n_input <= 0 || n_input > 100000) // check valid - based on coursework brief
    {
        std::cerr << "Error: n must be positive and at most 100000\n";
        return 1;
    }
    const std::size_t n = static_cast<std::size_t>(n_input); // cast to std::size_t

    const int repeats = std::atoi(argv[2]); // read repeats and ensure its positive
    if (repeats <= 0)
    {
        std::cerr << "Error: repeats must be positive\n";
        return 1;
    }

    const std::filesystem::path raw_csv_path(argv[3]);    // where to save the CSV
    const std::filesystem::path plot_output_dir(argv[4]); // where to save the plots

    if (raw_csv_path.has_parent_path()) // check whether the CSV path includes a parent directory
    {
        std::filesystem::create_directories(
            raw_csv_path.parent_path()); // Create the folder if it doesnt already exist
    }
    std::filesystem::create_directories(
        plot_output_dir); // create the plot output directory as well

    // Create an empty vector to store all block sizes given on the cmd line
    bool run_warmup = true;
    std::vector<int> block_sizes;
    block_sizes.reserve(static_cast<std::size_t>(argc - 5)); // reserve enough space in advance

    for (int argi = 5; argi < argc; ++argi) // read block sizes from the cmd line
    {
        if (parse_warmup_option(argv[argi], run_warmup))
        {
            continue;
        }

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

    if (run_warmup)
    {
        {
            // Make a copy of the original matrix called warmup - we make a copy because Cholesky
            // modifes the matrix
            std::vector<double> warmup_matrix = original_matrix;
            const double warmup_elapsed = run_baseline(warmup_matrix.data(), n);
            if (warmup_elapsed < 0.0)
            {
                std::cerr << "Error: baseline warm-up failed\n";
                return 1;
            }
        }
    }

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
    for (const std::string& optimisation : blocked_optimisations())
    {
        // Loop over block sizes
        for (const int block_size : block_sizes)
        {
            if (run_warmup)
            {
                {
                    // Make copy, run optimisation variant warm up with checking
                    std::vector<double> warmup_matrix = original_matrix;
                    const double warmup_elapsed =
                        run_blocked_variant(optimisation, warmup_matrix.data(), n, block_size);
                    if (warmup_elapsed < 0.0)
                    {
                        std::cerr << "Error: warm-up failed for " << optimisation
                                  << " with block_size=" << block_size << '\n';
                        return 1;
                    }
                }
            }

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

    // Construct file path to python plotting script
    const std::filesystem::path plot_script_path =
        std::filesystem::path(MPHIL_PROJECT_SOURCE_DIR) / "plot" / "plot_block_size_metrics.py";

    // build the python running cmd
    const std::string plot_command = "python3 " + quoted_path(plot_script_path) + " " +
        quoted_path(raw_csv_path) + " " + quoted_path(plot_output_dir);

    // run the cmd
    const int plot_status = std::system(plot_command.c_str());
    if (plot_status != 0)
    {
        std::cerr << "Error: plotting command failed with status " << plot_status << '\n';
        return 1;
    }

    // print stuff
    std::cout << "raw_csv=" << raw_csv_path << '\n';
    std::cout << "plot_output_dir=" << plot_output_dir << '\n';

    return 0;
}
