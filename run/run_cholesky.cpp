/**
 * @file run_cholesky.cpp
 * @brief Command-line entry point for the benchmark modes.
 */

#include "perf_modes.h"

#include <iostream>
#include <string>

namespace
{
/**
 * @brief Prints the supported benchmark modes and their argument shapes.
 * @param program_name Executable name used in the usage banner.
 */
void print_usage(const char* program_name)
{
    std::cerr << "Usage:\n"
              << "  " << program_name << " matrix-generator-compare [n] [raw_csv]\n"
              << "  " << program_name
              << " method-compare <n> <repeats> <raw_csv> [methods...] [--threads N] "
                 "[--block-size N] [--block-size-for METHOD=SIZE ...] [--correctness]\n"
              << "  " << program_name
              << " matrix-size-sweep <optimisation> <repeats> <raw_csv> <n1> [n2 ...] "
                 "[--threads N] [--block-size N] [--correctness]\n"
              << "  " << program_name
              << " block-size-sweep <optimisation> <n> <repeats> <raw_csv> <block_size1> "
                 "[block_size2 ...] [--threads N] [--correctness]\n"
              << "  " << program_name
              << " thread-count-sweep <n> <repeats> <raw_csv> --threads <t1> [t2 ...] "
                 "[--methods <m1> [m2 ...]] [--block-size N] [--correctness]\n"
              << "Notes:\n"
              << "  OpenMP benchmark methods require explicit --threads input.\n"
              << "  Blocked benchmark methods require explicit block-size input.\n";
}
} // namespace

/**
 * @brief Dispatches the benchmark CLI to the selected mode implementation.
 * @param argc Raw process argument count.
 * @param argv Raw process argument vector.
 * @return Process-style exit code from the selected mode.
 */
int main(int argc, char* argv[])
{
    if (argc < 2)
    {
        print_usage(argv[0]);
        return 1;
    }

    const std::string mode(argv[1]);

    if (mode == "fixed-size" || mode == "fixed_size" || mode == "method-compare" ||
        mode == "method-comparison" || mode == "method_compare" || mode == "method_comparison")
    {
        return run_fixed_size_comparison_mode(argc - 1, argv + 1);
    }

    if (mode == "matrix-generator-compare" || mode == "matrix_generator_compare" ||
        mode == "generator-compare" || mode == "generator_compare")
    {
        return run_matrix_generator_comparison_mode(argc - 1, argv + 1);
    }

    if (mode == "scaling" || mode == "matrix-size-sweep" || mode == "matrix_size_sweep")
    {
        return run_scaling_mode(argc - 1, argv + 1);
    }

    if (mode == "block-size-sweep" || mode == "block_size_sweep")
    {
        return run_block_size_sweep_mode(argc - 1, argv + 1);
    }

    if (mode == "thread-count-sweep" || mode == "thread_count_sweep")
    {
        return run_thread_count_sweep_mode(argc - 1, argv + 1);
    }

    if (mode == "help" || mode == "--help" || mode == "-h")
    {
        print_usage(argv[0]);
        return 0;
    }

    std::cerr << "Error: unknown mode '" << mode << "'\n";
    print_usage(argv[0]);
    return 1;
}
