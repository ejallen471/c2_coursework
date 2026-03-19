/**
 * @file perf_modes.h
 * @brief Declarations for the benchmark modes used by `run_cholesky`.
 */

#ifndef PERF_MODES_H
#define PERF_MODES_H

/**
 * @defgroup BenchmarkModes Benchmark Modes
 * @brief Entry points for the benchmark modes supported by `run_cholesky`.
 *
 * Each function handles one command-line mode. It parses that mode's arguments,
 * runs the requested benchmarks, and writes the CSV output used for plots and tests.
 * @{
 */

/**
 * @brief Run the fixed-size comparison mode.
 *
 * This is the implementation behind the `method-compare` command.
 *
 * @param argc Number of arguments for this mode.
 * @param argv Argument list for this mode.
 * @return Exit code in the usual process style.
 */
int run_fixed_size_comparison_mode(int argc, char* argv[]);

/**
 * @brief Run the matrix-generator comparison mode.
 *
 * This is the implementation behind the `matrix-generator-compare` command.
 *
 * @param argc Number of arguments for this mode.
 * @param argv Argument list for this mode.
 * @return Exit code in the usual process style.
 */
int run_matrix_generator_comparison_mode(int argc, char* argv[]);

/**
 * @brief Run the matrix-size sweep mode.
 *
 * This is the implementation behind the `matrix-size-sweep` command.
 *
 * @param argc Number of arguments for this mode.
 * @param argv Argument list for this mode.
 * @return Exit code in the usual process style.
 */
int run_scaling_mode(int argc, char* argv[]);

/**
 * @brief Run the block-size sweep mode.
 *
 * This is the implementation behind the `block-size-sweep` command.
 *
 * @param argc Number of arguments for this mode.
 * @param argv Argument list for this mode.
 * @return Exit code in the usual process style.
 */
int run_block_size_sweep_mode(int argc, char* argv[]);

/**
 * @brief Run the OpenMP thread-count sweep mode.
 *
 * This is the implementation behind the `thread-count-sweep` command.
 *
 * @param argc Number of arguments for this mode.
 * @param argv Argument list for this mode.
 * @return Exit code in the usual process style.
 */
int run_thread_count_sweep_mode(int argc, char* argv[]);

/** @} */

#endif
