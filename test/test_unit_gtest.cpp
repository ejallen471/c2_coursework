/**
 * @file test_unit_gtest.cpp
 * @brief GoogleTest unit coverage for matrix helpers, parsing, and small deterministic cases.
 */

#include "matrix.h"
#include "cholesky_decomposition.h"
#include "runtime_cholesky.h"
#include "test_suite_helpers.h"

#include <gtest/gtest.h>

#include <cmath>
#include <vector>

TEST(SmallCaseTest, MatchesKnownWorkedExamples)
{
    struct SmallCase
    {
        const char* name;
        std::vector<double> input;
        std::vector<double> expected;
        int n;
        CholeskyVersion version;
    };

    const std::vector<SmallCase> cases = {
        {"brief_2x2",
         {4.0, 2.0, 2.0, 26.0},
         {2.0, 1.0, 1.0, 5.0},
         2,
         CholeskyVersion::Baseline},
        {"spd_2x2",
         {9.0, 3.0, 3.0, 5.0},
         {3.0, 1.0, 1.0, 2.0},
         2,
         CholeskyVersion::Baseline},
        {"spd_3x3",
         {4.0, 2.0, 2.0, 2.0, 10.0, 5.0, 2.0, 5.0, 9.0},
         {2.0, 1.0, 1.0, 1.0, 3.0, 4.0 / 3.0, 1.0, 4.0 / 3.0, std::sqrt(56.0 / 9.0)},
         3,
         CholeskyVersion::Baseline},
        {"lower_triangle_runtime_selected",
         {9.0, 3.0, 3.0, 5.0},
         {3.0, 1.0, 1.0, 2.0},
         2,
         CholeskyVersion::LowerTriangleOnly},
    };

    for (const SmallCase& small_case : cases)
    {
        SCOPED_TRACE(small_case.name);
        std::vector<double> factorised = small_case.input;
        const double elapsed = timed_cholesky_factorisation_versioned(
            factorised.data(), small_case.n, small_case.version);

        ASSERT_GE(elapsed, 0.0);
        ASSERT_EQ(factorised.size(), small_case.expected.size());
        for (std::size_t index = 0; index < factorised.size(); ++index)
        {
            EXPECT_NEAR(factorised[index], small_case.expected[index], 1.0e-12)
                << "at index " << index;
        }
    }
}

TEST(InputValidationTest, RejectsInvalidInput)
{
    EXPECT_LT(timed_cholesky_factorisation(nullptr, 2), 0.0);

    double identity[4] = {1.0, 0.0, 0.0, 1.0};
    EXPECT_LT(timed_cholesky_factorisation(identity, 0), 0.0);

    double tiny[1] = {1.0};
    EXPECT_LT(timed_cholesky_factorisation(tiny, 100001), 0.0);
}

TEST(MatrixBasicsTest, IdentityFactorisesToIdentity)
{
    std::vector<double> factorised = make_identity_matrix(5);
    const std::vector<double> expected = make_identity_matrix(5);

    ASSERT_GE(timed_cholesky_factorisation(factorised.data(), 5), 0.0);
    EXPECT_TRUE(vectors_close(factorised, expected, 1.0e-12, 1.0e-12));
}

TEST(MatrixBasicsTest, DiagonalMatrixFactorisesExactly)
{
    const std::vector<double> diagonal = {4.0, 9.0, 16.0, 25.0};
    std::vector<double> factorised = make_diagonal_matrix(diagonal);
    const std::vector<double> expected = {
        2.0, 0.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 0.0, 5.0};

    ASSERT_GE(timed_cholesky_factorisation(factorised.data(), 4), 0.0);
    EXPECT_TRUE(vectors_close(factorised, expected, 1.0e-12, 1.0e-12));
}

TEST(LogDetTest, RecoversWorkedExampleAndDiagonalExample)
{
    {
        std::vector<double> factorised = make_brief_example_matrix();
        ASSERT_GE(timed_cholesky_factorisation(factorised.data(), 2), 0.0);
        EXPECT_NEAR(logdet_from_factorised_storage(factorised, 2), std::log(100.0), 1.0e-12);
    }

    {
        const std::vector<double> diagonal = {4.0, 9.0, 16.0};
        std::vector<double> factorised = make_diagonal_matrix(diagonal);
        ASSERT_GE(timed_cholesky_factorisation(factorised.data(), 3), 0.0);
        EXPECT_NEAR(logdet_from_factorised_storage(factorised, 3), std::log(4.0 * 9.0 * 16.0), 1.0e-12);
    }
}

TEST(GeneratedSpdTest, GeneratedMatricesSatisfySpdChecksAndReconstruct)
{
    for (const int n : {4, 8, 16, 32})
    {
        SCOPED_TRACE(n);
        const std::vector<double> original = make_generated_spd_matrix(n);
        ASSERT_TRUE(matrix_satisfies_generated_spd_conditions(original, n));

        std::vector<double> factorised = original;
        ASSERT_GE(timed_cholesky_factorisation(factorised.data(), n), 0.0);

        const std::vector<double> reconstructed = reconstruct_from_factorised_storage(factorised, n);
        EXPECT_TRUE(vectors_close(reconstructed, original, 1.0e-9, 1.0e-9));
    }
}

TEST(CourseworkBriefMatrixTest, BriefAndAdjustedMatricesFactorise)
{
    for (const int n : {4, 8, 16, 32})
    {
        SCOPED_TRACE(n);
        const std::vector<double> original = make_coursework_brief_matrix(n);
        const std::vector<double> adjusted = make_gershgorin_adjusted_copy(original, n);

        ASSERT_EQ(
            original.size(),
            static_cast<std::size_t>(n) * static_cast<std::size_t>(n));
        ASSERT_TRUE(matrix_is_symmetric(original, n, 1.0e-12, 1.0e-12));
        ASSERT_TRUE(diagonal_is_positive(original, n));
        ASSERT_TRUE(matrix_satisfies_generated_spd_conditions(adjusted, n));

        std::vector<double> factorised = original;
        ASSERT_GE(timed_cholesky_factorisation(factorised.data(), n), 0.0);
        EXPECT_TRUE(vectors_close(
            reconstruct_from_factorised_storage(factorised, n), original, 1.0e-9, 1.0e-9));

        std::vector<double> adjusted_factorised = adjusted;
        ASSERT_GE(timed_cholesky_factorisation(adjusted_factorised.data(), n), 0.0);
        EXPECT_TRUE(vectors_close(
            reconstruct_from_factorised_storage(adjusted_factorised, n), adjusted, 1.0e-9, 1.0e-9));
    }
}

TEST(RepeatabilityTest, BaselineProducesRepeatableResults)
{
    const int n = 10;
    const std::vector<double> original = make_generated_spd_matrix(n);

    std::vector<double> reference = original;
    ASSERT_GE(timed_cholesky_factorisation(reference.data(), n), 0.0);

    for (int run = 0; run < 5; ++run)
    {
        SCOPED_TRACE(run);
        std::vector<double> current = original;
        ASSERT_GE(timed_cholesky_factorisation(current.data(), n), 0.0);
        EXPECT_TRUE(vectors_close(current, reference, 1.0e-12, 1.0e-12));
    }
}

TEST(MatrixGenerationTest, InvalidSizesReturnEmptyMatrices)
{
    EXPECT_TRUE(make_generated_spd_matrix(0).empty());
    EXPECT_TRUE(make_generated_spd_matrix(-3).empty());
}

TEST(MatrixGenerationTest, OptionsPreserveSpdAndSeedControlsReproducibility)
{
    MatrixGenerationOptions options;
    options.seed = 20260310ULL;
    options.amplitude = 1.0;
    options.nugget = 1.0e-3;

    for (const int n : {1, 4, 12})
    {
        SCOPED_TRACE(n);
        const std::vector<double> original = make_generated_spd_matrix(n, options);
        ASSERT_EQ(
            original.size(),
            static_cast<std::size_t>(n) * static_cast<std::size_t>(n));
        ASSERT_TRUE(matrix_satisfies_generated_spd_conditions(original, n));

        std::vector<double> factorised = original;
        ASSERT_GE(run_cholesky_version(factorised.data(), n, CholeskyVersion::Baseline), 0.0);
        EXPECT_TRUE(vectors_close(
            reconstruct_from_factorised_storage(factorised, n), original, 1.0e-9, 1.0e-9));
    }

    options.seed = 1234ULL;
    const std::vector<double> first = make_generated_spd_matrix(10, options);
    const std::vector<double> second = make_generated_spd_matrix(10, options);
    EXPECT_TRUE(vectors_close(first, second, 0.0, 0.0));

    options.seed = 5678ULL;
    const std::vector<double> third = make_generated_spd_matrix(10, options);
    EXPECT_FALSE(vectors_close(first, third, 0.0, 0.0));
}
