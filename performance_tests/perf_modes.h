/**
 * @file perf_modes.h
 * @brief Entry points for the benchmark modes exposed by `run_cholesky`.
 */

#ifndef PERF_MODES_H
#define PERF_MODES_H

/**
 * @brief Runs the single-execution benchmark mode.
 * @param argc Argument count for the mode-specific argv slice.
 * @param argv Mode-specific arguments.
 * @return Process-style exit code.
 */
int run_time_mode(int argc, char* argv[]);

/**
 * @brief Runs the fixed-size comparison benchmark mode.
 * @param argc Argument count for the mode-specific argv slice.
 * @param argv Mode-specific arguments.
 * @return Process-style exit code.
 */
int run_fixed_size_comparison_mode(int argc, char* argv[]);

/**
 * @brief Runs the matrix-scaling benchmark mode.
 * @param argc Argument count for the mode-specific argv slice.
 * @param argv Mode-specific arguments.
 * @return Process-style exit code.
 */
int run_scaling_mode(int argc, char* argv[]);

/**
 * @brief Runs the block-size sweep benchmark mode.
 * @param argc Argument count for the mode-specific argv slice.
 * @param argv Mode-specific arguments.
 * @return Process-style exit code.
 */
int run_block_size_sweep_mode(int argc, char* argv[]);

#endif
