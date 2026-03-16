/**
 * @file test_coursework_brief_matrix.cpp
 * @brief Verifies the coursework brief matrix and its Gershgorin-adjusted copy factorise correctly.
 */

#include "mphil_dis_cholesky.h"
#include "test_helpers.h"

#include <iostream>
#include <vector>

int main()
{
    const int sizes[] = {4, 8, 16, 32};

    for (const int n : sizes)
    {
        const std::vector<double> original = make_coursework_brief_matrix(n);
        const std::vector<double> gershgorin_adjusted = make_gershgorin_adjusted_copy(original, n);

        if (original.size() != static_cast<std::size_t>(n) * static_cast<std::size_t>(n))
        {
            std::cerr << "test_coursework_brief_matrix failed: unexpected storage size for n = "
                      << n << '\n';
            return 1;
        }

        if (!matrix_is_symmetric(original, n, 1e-12, 1e-12))
        {
            std::cerr << "test_coursework_brief_matrix failed: matrix was not symmetric for n = "
                      << n << '\n';
            return 1;
        }

        if (!diagonal_is_positive(original, n))
        {
            std::cerr
                << "test_coursework_brief_matrix failed: matrix diagonal was not positive for n = "
                << n << '\n';
            return 1;
        }

        if (!matrix_satisfies_generated_spd_conditions(gershgorin_adjusted, n))
        {
            std::cerr << "test_coursework_brief_matrix failed: Gershgorin-adjusted copy did not "
                         "satisfy the SPD construction checks for n = "
                      << n << '\n';
            return 1;
        }

        std::vector<double> factorised = original;
        const double elapsed = timed_cholesky_factorisation(factorised.data(), n);
        if (elapsed < 0.0)
        {
            std::cerr << "test_coursework_brief_matrix failed: baseline factorisation returned "
                      << elapsed << " for n = " << n << '\n';
            return 1;
        }

        const std::vector<double> reconstructed = reconstruct_from_factorised_storage(factorised, n);
        if (!vectors_close(reconstructed, original, 1e-9, 1e-9))
        {
            std::cerr << "test_coursework_brief_matrix failed: reconstruction mismatch for n = "
                      << n << '\n';
            return 1;
        }

        std::vector<double> factorised_adjusted = gershgorin_adjusted;
        const double adjusted_elapsed = timed_cholesky_factorisation(factorised_adjusted.data(), n);
        if (adjusted_elapsed < 0.0)
        {
            std::cerr << "test_coursework_brief_matrix failed: Gershgorin-adjusted factorisation "
                         "returned "
                      << adjusted_elapsed << " for n = " << n << '\n';
            return 1;
        }

        const std::vector<double> reconstructed_adjusted =
            reconstruct_from_factorised_storage(factorised_adjusted, n);
        if (!vectors_close(reconstructed_adjusted, gershgorin_adjusted, 1e-9, 1e-9))
        {
            std::cerr << "test_coursework_brief_matrix failed: Gershgorin-adjusted reconstruction "
                         "mismatch for n = "
                      << n << '\n';
            return 1;
        }
    }

    std::cout << "test_coursework_brief_matrix passed\n";
    return 0;
}
