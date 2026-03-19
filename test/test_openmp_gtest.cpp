/**
 * @file test_openmp_gtest.cpp
 * @brief GoogleTest OpenMP robustness and historical regression coverage.
 */

#include "matrix.h"
#include "test_suite_helpers.h"

#include <gtest/gtest.h>

#include <string>
#include <vector>

TEST(OpenMPSuite, MethodsRemainCorrectAcrossSupportedThreadCounts)
{
    const int n = 64;
    const std::vector<double> original = make_generated_spd_matrix(n);
    LogDetValue reference_logdet = 0.0L;
    ASSERT_TRUE(reference_logdet_for_matrix(original, n, reference_logdet));

    // Sweep a conservative set of thread counts so the OpenMP tests stay portable across
    // laptops and CI runners while still checking that correctness is thread-count invariant.
    const std::vector<int> thread_counts = supported_thread_counts({1, 2, 4, 8});

    for (const CholeskyVersion version : openmp_test_versions())
    {
        SCOPED_TRACE(optimisation_name(version));
        bool first_run_seen = false;
        LogDetValue first_run_logdet = 0.0L;

        for (const int thread_count : thread_counts)
        {
            SCOPED_TRACE(thread_count);
            // Reconfigure the OpenMP runtime before each run group so any mismatch between
            // requested and actual thread counts shows up as a test failure immediately.
            set_test_openmp_thread_count(thread_count);

            for (int repeat = 0; repeat < 2; ++repeat)
            {
                SCOPED_TRACE(repeat);
                std::vector<double> factorised = original;
                ASSERT_GE(
                    run_version_with_block_size(
                        factorised, n, version, kDefaultBlockedCholeskyBlockSize),
                    0.0);

                FactorisationMetrics metrics;
                std::string failure;
                ASSERT_TRUE(validate_factorisation(
                    original, factorised, n, reference_logdet, failure, &metrics))
                    << failure;

                if (!first_run_seen)
                {
                    first_run_seen = true;
                    first_run_logdet = metrics.logdet_factor;
                }
                else
                {
                    EXPECT_TRUE(long_double_nearly_equal(metrics.logdet_factor, first_run_logdet))
                        << "first=" << static_cast<double>(first_run_logdet)
                        << " current=" << static_cast<double>(metrics.logdet_factor);
                }
            }
        }
    }
}

TEST(OpenMPSuite, FocusMethodsRemainConsistentAcrossThreadCountsAndBlockSizes)
{
    const int n = 32;
    const std::vector<double> original = make_generated_spd_matrix(n);
    LogDetValue reference_logdet = 0.0L;
    ASSERT_TRUE(reference_logdet_for_matrix(original, n, reference_logdet));

    const std::vector<int> focus_block_sizes = {3, 7, 16, n, n + 1};
    // Reuse the same portable thread-count grid here so block-size checks also cover several
    // OpenMP team sizes without assuming a large workstation is available.
    const std::vector<int> thread_counts = supported_thread_counts({1, 2, 4, 8});

    for (const CholeskyVersion version : blocked_openmp_test_versions())
    {
        SCOPED_TRACE(optimisation_name(version));

        for (const int block_size : focus_block_sizes)
        {
            SCOPED_TRACE(block_size);
            std::vector<double> factorised = original;
            ASSERT_GE(run_version_with_block_size(factorised, n, version, block_size), 0.0);

            std::string failure;
            EXPECT_TRUE(validate_factorisation(
                original, factorised, n, reference_logdet, failure, nullptr))
                << failure;
        }

        for (const int thread_count : thread_counts)
        {
            SCOPED_TRACE(thread_count);
            // Reset the OpenMP runtime before each correctness check so this test exercises
            // the specific thread count named by the current parameter.
            set_test_openmp_thread_count(thread_count);

            std::vector<double> factorised = original;
            ASSERT_GE(run_version_with_block_size(factorised, n, version, 7), 0.0);

            std::string failure;
            EXPECT_TRUE(validate_factorisation(
                original, factorised, n, reference_logdet, failure, nullptr))
                << failure;
        }
    }
}

TEST(OpenMPSuite, AwkwardBlockedRunsDoNotProduceNanOrInf)
{
    const int n = 48;
    const int awkward_block_size = 5;
    const std::vector<double> original = make_generated_spd_matrix(n);
    LogDetValue reference_logdet = 0.0L;
    ASSERT_TRUE(reference_logdet_for_matrix(original, n, reference_logdet));

    for (const CholeskyVersion version : openmp_test_versions())
    {
        SCOPED_TRACE(optimisation_name(version));
        for (const int thread_count : supported_thread_counts({1, 2, 4}))
        {
            SCOPED_TRACE(thread_count);
            // Force the OpenMP runtime onto the current thread-count case so the historical
            // NaN/Inf regression is checked under more than one level of parallelism.
            set_test_openmp_thread_count(thread_count);

            std::vector<double> factorised = original;
            ASSERT_GE(run_version_with_block_size(
                          factorised, n, version, awkward_block_size),
                      0.0);

            std::string failure;
            EXPECT_TRUE(validate_factorisation(
                original, factorised, n, reference_logdet, failure, nullptr))
                << failure;
        }
    }
}
