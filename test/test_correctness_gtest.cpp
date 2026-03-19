/**
 * @file test_correctness_gtest.cpp
 * @brief GoogleTest correctness coverage for mathematical validity and cross-method consistency.
 */

#include "matrix.h"
#include "test_suite_helpers.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <string>
#include <vector>

TEST(CorrectnessSuite, AllMethodsProduceValidFactorisations)
{
    const std::vector<int> sizes = {1, 2, 4, 8, 16, 32, 64};

    for (const int n : sizes)
    {
        SCOPED_TRACE(n);
        const std::vector<double> original = make_generated_spd_matrix(n);
        LogDetValue reference_logdet = 0.0L;
        ASSERT_TRUE(reference_logdet_for_matrix(original, n, reference_logdet));

        for (const CholeskyVersion version : all_test_versions())
        {
            SCOPED_TRACE(optimisation_name(version));
            std::vector<double> factorised = original;
            const double elapsed =
                run_version_with_block_size(factorised, n, version, kDefaultBlockedCholeskyBlockSize);

            ASSERT_GE(elapsed, 0.0);

            FactorisationMetrics metrics;
            metrics.elapsed_seconds = elapsed;
            std::string failure;
            EXPECT_TRUE(validate_factorisation(
                original, factorised, n, reference_logdet, failure, &metrics))
                << failure;
        }
    }
}

TEST(CorrectnessSuite, MethodsRemainConsistentOnLogDetAndReconstruction)
{
    for (const int n : {16, 64})
    {
        SCOPED_TRACE(n);
        const std::vector<double> original = make_generated_spd_matrix(n);
        LogDetValue reference_logdet = 0.0L;
        ASSERT_TRUE(reference_logdet_for_matrix(original, n, reference_logdet));

        std::vector<double> baseline = original;
        ASSERT_GE(run_version_default(baseline, n, CholeskyVersion::Baseline), 0.0);
        const LogDetValue baseline_logdet =
            logdet_from_factorised_storage(baseline, static_cast<std::size_t>(n));

        for (const CholeskyVersion version : all_test_versions())
        {
            SCOPED_TRACE(optimisation_name(version));
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
            EXPECT_TRUE(long_double_nearly_equal(metrics.logdet_factor, baseline_logdet))
                << "baseline=" << static_cast<double>(baseline_logdet)
                << " actual=" << static_cast<double>(metrics.logdet_factor);
        }
    }
}

TEST(CorrectnessSuite, BlockedMethodsHandleAwkwardBlockSizes)
{
    for (const int n : {32, 64})
    {
        SCOPED_TRACE(n);
        const std::vector<double> original = make_generated_spd_matrix(n);
        LogDetValue reference_logdet = 0.0L;
        ASSERT_TRUE(reference_logdet_for_matrix(original, n, reference_logdet));

        std::vector<int> block_sizes = {1, 2, 3, 4, 5, 7, 8, 10, 16, 32, n, n + 1};
        block_sizes.erase(std::unique(block_sizes.begin(), block_sizes.end()), block_sizes.end());

        for (const CholeskyVersion version : blocked_test_versions())
        {
            SCOPED_TRACE(optimisation_name(version));
            for (const int block_size : block_sizes)
            {
                SCOPED_TRACE(block_size);
                std::vector<double> factorised = original;
                ASSERT_GE(run_version_with_block_size(factorised, n, version, block_size), 0.0);

                std::string failure;
                EXPECT_TRUE(validate_factorisation(
                    original, factorised, n, reference_logdet, failure, nullptr))
                    << failure;
            }
        }
    }
}
