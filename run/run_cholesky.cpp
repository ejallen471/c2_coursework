#include "perf_modes.h"

#include <iostream>
#include <string>

namespace
{
void print_usage(const char* program_name)
{
    std::cerr << "Usage:\n"
              << "  " << program_name << " time <optimisation> <n> [raw_csv]\n"
              << "  " << program_name
              << " fixed-size <n> <repeats> <raw_csv>\n"
              << "  " << program_name
              << " scaling <optimisation> <repeats> <raw_csv> <n1> [n2 ...]\n"
              << "  " << program_name
              << " block-size-sweep <optimisation> <n> <repeats> <raw_csv> <block_size1> [block_size2 ...]\n";
}
} // namespace

int main(int argc, char* argv[])
{
    if (argc < 2)
    {
        print_usage(argv[0]);
        return 1;
    }

    const std::string mode(argv[1]);

    if (mode == "time")
    {
        return run_time_mode(argc - 1, argv + 1);
    }

    if (mode == "fixed-size" || mode == "fixed_size")
    {
        return run_fixed_size_comparison_mode(argc - 1, argv + 1);
    }

    if (mode == "scaling")
    {
        return run_scaling_mode(argc - 1, argv + 1);
    }

    if (mode == "block-size-sweep" || mode == "block_size_sweep")
    {
        return run_block_size_sweep_mode(argc - 1, argv + 1);
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
