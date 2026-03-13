#include "matrix.h"
#include "runtime_cholesky.h"
#include "test_helpers.h"

#include <iostream>
#include <vector>

namespace
{
bool check_version_against_baseline(CholeskyVersion version, const std::vector<double>& original,
                                    const std::vector<double>& baseline_factor, int n)
{
    std::vector<double> current = original;
    const double elapsed = run_cholesky_version(current.data(), n, version);

    if (elapsed < 0.0)
    {
        std::cerr << "test_version_correctness failed: version " << optimisation_name(version)
                  << " returned " << elapsed << " for n=" << n << '\n';
        return false;
    }

    if (!matrix_is_symmetric(current, n, 1e-10, 1e-10))
    {
        std::cerr << "test_version_correctness failed: version " << optimisation_name(version)
                  << " did not preserve mirrored storage for n=" << n << '\n';
        return false;
    }

    if (!diagonal_is_positive(current, n))
    {
        std::cerr << "test_version_correctness failed: version " << optimisation_name(version)
                  << " produced a non-positive diagonal for n=" << n << '\n';
        return false;
    }

    const std::vector<double> reconstructed = reconstruct_from_factorised_storage(current, n);
    if (!vectors_close(reconstructed, original, 1e-8, 1e-8))
    {
        std::cerr << "test_version_correctness failed: reconstruction mismatch for version "
                  << optimisation_name(version) << " and n=" << n << '\n';
        return false;
    }

    const double expected_logdet = logdet_from_factorised_storage(baseline_factor, n);
    const double actual_logdet = logdet_from_factorised_storage(current, n);
    if (!nearly_equal(actual_logdet, expected_logdet, 1e-9, 1e-9))
    {
        std::cerr << "test_version_correctness failed: logdet mismatch for version "
                  << optimisation_name(version) << " and n=" << n << '\n';
        return false;
    }

    if (!vectors_close(current, baseline_factor, 1e-8, 1e-8))
    {
        std::cerr << "test_version_correctness failed: factor output differs from baseline for version "
                  << optimisation_name(version) << " and n=" << n << '\n';
        return false;
    }

    return true;
}
} // namespace

int main()
{
    const CholeskyVersion versions[] = {CholeskyVersion::Baseline,
                                        CholeskyVersion::LowerTriangleOnly,
                                        CholeskyVersion::InlineMirror,
                                        CholeskyVersion::LoopCleanup,
                                        CholeskyVersion::AccessPatternAware,
                                        CholeskyVersion::CacheBlocked,
                                        CholeskyVersion::VectorFriendly,
                                        CholeskyVersion::BlockedVectorised};

    MatrixGenerationOptions options;
    options.seed = 20260310ULL;

    const int sizes[] = {4, 8, 16};
    for (const int n : sizes)
    {
        const std::vector<double> original = make_generated_spd_matrix(n, options);
        std::vector<double> baseline_factor = original;

        const double baseline_elapsed =
            run_cholesky_version(baseline_factor.data(), n, CholeskyVersion::Baseline);
        if (baseline_elapsed < 0.0)
        {
            std::cerr << "test_version_correctness failed: baseline returned " << baseline_elapsed
                      << " for n=" << n << '\n';
            return 1;
        }

        for (const CholeskyVersion version : versions)
        {
            if (!check_version_against_baseline(version, original, baseline_factor, n))
            {
                return 1;
            }

            CholeskyVersion parsed;
            if (!parse_optimisation_name(optimisation_name(version), parsed) || parsed != version)
            {
                std::cerr << "test_version_correctness failed: name round-trip failed for "
                          << optimisation_name(version) << '\n';
                return 1;
            }
        }
    }

    {
        const int n = 8;
        const std::vector<double> original = make_generated_spd_matrix(n, options);
        std::vector<double> baseline_factor = original;

        const double baseline_elapsed =
            run_cholesky_version(baseline_factor.data(), n, CholeskyVersion::Baseline);
        if (baseline_elapsed < 0.0)
        {
            std::cerr << "test_version_correctness failed: baseline returned " << baseline_elapsed
                      << " for OpenMP comparison n=" << n << '\n';
            return 1;
        }

#if defined(MPHIL_HAVE_OPENMP) && MPHIL_HAVE_OPENMP
        if (!check_version_against_baseline(CholeskyVersion::OpenMP1, original, baseline_factor, n))
        {
            return 1;
        }
#else
        std::vector<double> current = original;
        const double elapsed = run_cholesky_version(current.data(), n, CholeskyVersion::OpenMP1);
        if (elapsed != -4.0)
        {
            std::cerr << "test_version_correctness failed: expected placeholder return code -4 for "
                      << optimisation_name(CholeskyVersion::OpenMP1) << ", got " << elapsed
                      << '\n';
            return 1;
        }
#endif
    }

    {
        std::vector<double> matrix = make_identity_matrix(2);
        const CholeskyVersion openmp_versions[] = {CholeskyVersion::OpenMP2,
                                                   CholeskyVersion::OpenMP3};

        for (const CholeskyVersion version : openmp_versions)
        {
            std::vector<double> current = matrix;
            const double elapsed = run_cholesky_version(current.data(), 2, version);

            if (elapsed != -4.0)
            {
                std::cerr << "test_version_correctness failed: expected placeholder return code -4 for "
                          << optimisation_name(version) << ", got " << elapsed << '\n';
                return 1;
            }
        }
    }

    std::cout << "test_version_correctness passed\n";
    return 0;
}
