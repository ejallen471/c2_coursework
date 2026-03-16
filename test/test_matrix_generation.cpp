/**
 * @file test_matrix_generation.cpp
 * @brief Validates matrix generator options, reproducibility, and SPD preservation.
 */

#include "matrix.h"
#include "runtime_cholesky.h"
#include "test_helpers.h"

#include <cmath>
#include <iostream>
#include <vector>

namespace
{
bool generator_settings_preserve_spd(const MatrixGenerationOptions& options, int n)
{
    const std::vector<double> original = make_generated_spd_matrix(n, options);

    if (original.size() != static_cast<std::size_t>(n) * static_cast<std::size_t>(n))
    {
        std::cerr << "test_matrix_generation failed: unexpected storage size for n=" << n << '\n';
        return false;
    }

    if (!matrix_satisfies_generated_spd_conditions(original, n))
    {
        std::cerr << "test_matrix_generation failed: generated matrix did not satisfy the SPD construction checks for n="
                  << n << '\n';
        return false;
    }

    std::vector<double> factorised = original;
    const double elapsed = run_cholesky_version(factorised.data(), n, CholeskyVersion::Baseline);
    if (elapsed < 0.0)
    {
        std::cerr << "test_matrix_generation failed: baseline factorisation returned " << elapsed
                  << " for n=" << n << '\n';
        return false;
    }

    const std::vector<double> reconstructed = reconstruct_from_factorised_storage(factorised, n);
    if (!vectors_close(reconstructed, original, 1e-9, 1e-9))
    {
        std::cerr << "test_matrix_generation failed: reconstructed matrix does not match original for n="
                  << n << '\n';
        return false;
    }

    return true;
}
} // namespace

int main()
{
    if (!make_generated_spd_matrix(0).empty())
    {
        std::cerr << "test_matrix_generation failed: n = 0 should produce an empty matrix\n";
        return 1;
    }

    if (!make_generated_spd_matrix(-3).empty())
    {
        std::cerr << "test_matrix_generation failed: negative n should produce an empty matrix\n";
        return 1;
    }

    const int sizes[] = {1, 4, 12};

    {
        MatrixGenerationOptions options;
        options.seed = 20260310ULL;
        options.amplitude = 1.0;
        options.nugget = 1.0e-3;

        for (const int n : sizes)
        {
            if (!generator_settings_preserve_spd(options, n))
            {
                return 1;
            }
        }
    }

    {
        MatrixGenerationOptions options;
        options.seed = 1234ULL;

        const std::vector<double> first = make_generated_spd_matrix(10, options);
        const std::vector<double> second = make_generated_spd_matrix(10, options);

        if (!vectors_close(first, second, 0.0, 0.0))
        {
            std::cerr << "test_matrix_generation failed: strict diagonal dominance generator should be reproducible for identical seeds\n";
            return 1;
        }

        options.seed = 5678ULL;
        const std::vector<double> third = make_generated_spd_matrix(10, options);

        if (vectors_close(first, third, 0.0, 0.0))
        {
            std::cerr << "test_matrix_generation failed: strict diagonal dominance generator should change with the seed\n";
            return 1;
        }
    }

    std::cout << "test_matrix_generation passed\n";
    return 0;
}
