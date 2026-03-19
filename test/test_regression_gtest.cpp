/**
 * @file test_regression_gtest.cpp
 * @brief GoogleTest regression coverage for runtime method parsing and alias handling.
 */

#include "runtime_cholesky.h"
#include "test_suite_helpers.h"

#include <gtest/gtest.h>

#include <string>
#include <utility>
#include <vector>

TEST(RegressionSuite, MethodAliasesContinueToParse)
{
    const std::vector<std::pair<std::string, CholeskyVersion>> valid_cases = {
        {"baseline", CholeskyVersion::Baseline},
        {"lower_triangle", CholeskyVersion::LowerTriangleOnly},
        {"lowertriangle", CholeskyVersion::LowerTriangleOnly},
        {"lower_triangle_only", CholeskyVersion::LowerTriangleOnly},
        {"upper_triangle", CholeskyVersion::UpperTriangle},
        {"uppertriangle", CholeskyVersion::UpperTriangle},
        {"triangular_upper", CholeskyVersion::UpperTriangle},
        {"contiguous_access", CholeskyVersion::ContiguousAccess},
        {"contiguousaccess", CholeskyVersion::ContiguousAccess},
        {"access_pattern", CholeskyVersion::ContiguousAccess},
        {"cholesky_blocked_tile_kernels", CholeskyVersion::BlockedTileKernels},
        {"choleskyblockedtilekernels", CholeskyVersion::BlockedTileKernels},
        {"cholesky_blocked_tile_kernels_unrolled", CholeskyVersion::BlockedTileKernelsUnrolled},
        {"cholesky-blocked-tile-kernels-unrolled", CholeskyVersion::BlockedTileKernelsUnrolled},
        {"openmp_row_parallel_unblocked", CholeskyVersion::OpenMPRowParallelUnblocked},
        {"openmp_tile_parallel_blocked", CholeskyVersion::OpenMPTileParallelBlocked},
        {"openmp tile parallel blocked", CholeskyVersion::OpenMPTileParallelBlocked},
        {"openmp_block_row_parallel", CholeskyVersion::OpenMPBlockRowParallel},
        {"openmp_tile_list_parallel", CholeskyVersion::OpenMPTileListParallel},
        {"openmp_task_dag_blocked", CholeskyVersion::OpenMPTaskDAGBlocked},
    };

    for (const auto& valid_case : valid_cases)
    {
        SCOPED_TRACE(valid_case.first);
        CholeskyVersion parsed = CholeskyVersion::Baseline;
        ASSERT_TRUE(parse_optimisation_name(valid_case.first, parsed));
        EXPECT_EQ(parsed, valid_case.second);
    }
}

TEST(RegressionSuite, CanonicalNamesRoundTrip)
{
    for (const CholeskyVersion version : all_test_versions())
    {
        SCOPED_TRACE(optimisation_name(version));
        CholeskyVersion parsed = CholeskyVersion::Baseline;
        ASSERT_TRUE(parse_optimisation_name(optimisation_name(version), parsed));
        EXPECT_EQ(parsed, version);
    }
}

TEST(RegressionSuite, InvalidAliasesRemainRejected)
{
    CholeskyVersion parsed = CholeskyVersion::Baseline;
    EXPECT_FALSE(parse_optimisation_name("openmp6", parsed));
    EXPECT_FALSE(parse_optimisation_name("not_a_method", parsed));
}
