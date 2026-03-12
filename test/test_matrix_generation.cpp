#include "matrix.h"
#include "runtime_cholesky.h"
#include "test_helpers.h"

#include <iostream>
#include <vector>

namespace
{
bool generator_settings_preserve_spd(CovarianceKernel kernel, int n)
{
    MatrixGenerationOptions options;
    options.kernel = kernel;
    options.seed = 20260310ULL;
    options.randomize_points = false;
    options.nugget = 1.0e-3;

    const std::vector<double> original = make_generated_spd_matrix(n, options);

    if (original.size() != static_cast<std::size_t>(n) * static_cast<std::size_t>(n))
    {
        std::cerr << "test_matrix_generation failed: unexpected storage size for n=" << n << '\n';
        return false;
    }

    if (!matrix_is_symmetric(original, n, 0.0, 0.0))
    {
        std::cerr << "test_matrix_generation failed: generated matrix is not symmetric for n=" << n
                  << '\n';
        return false;
    }

    if (!diagonal_is_positive(original, n))
    {
        std::cerr << "test_matrix_generation failed: generated matrix has non-positive diagonal for n="
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

    const CovarianceKernel kernels[] = {CovarianceKernel::SquaredExponential,
                                        CovarianceKernel::Exponential,
                                        CovarianceKernel::RandomMixture};
    const int sizes[] = {1, 4, 12};

    for (const CovarianceKernel kernel : kernels)
    {
        for (const int n : sizes)
        {
            if (!generator_settings_preserve_spd(kernel, n))
            {
                return 1;
            }
        }
    }

    {
        MatrixGenerationOptions options;
        options.seed = 1234ULL;
        options.kernel = CovarianceKernel::RandomMixture;
        options.randomize_points = true;

        const std::vector<double> first = make_generated_spd_matrix(10, options);
        const std::vector<double> second = make_generated_spd_matrix(10, options);

        if (!vectors_close(first, second, 0.0, 0.0))
        {
            std::cerr << "test_matrix_generation failed: identical seeds should reproduce the same matrix\n";
            return 1;
        }

        options.seed = 5678ULL;
        const std::vector<double> third = make_generated_spd_matrix(10, options);

        if (vectors_close(first, third, 0.0, 0.0))
        {
            std::cerr << "test_matrix_generation failed: different seeds should change the generated matrix\n";
            return 1;
        }
    }

    std::cout << "test_matrix_generation passed\n";
    return 0;
}
