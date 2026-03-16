/**
 * @file test_generated_spd.cpp
 * @brief Validates generated SPD matrices and their reconstructions after factorisation.
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
        const std::vector<double> original = make_generated_spd_matrix(n);

        if (!matrix_satisfies_generated_spd_conditions(original, n))
        {
            std::cerr << "test_generated_spd failed SPD construction checks for n = " << n << '\n';
            return 1;
        }

        std::vector<double> c = original;

        const double elapsed = timed_cholesky_factorisation(c.data(), n);
        if (elapsed < 0.0)
        {
            std::cerr << "test_generated_spd failed for n = " << n << ": routine returned "
                      << elapsed << '\n';
            return 1;
        }

        const std::vector<double> reconstructed = reconstruct_from_factorised_storage(c, n);

        if (!vectors_close(reconstructed, original, 1e-9, 1e-9))
        {
            std::cerr << "test_generated_spd failed reconstruction for n = " << n << '\n';
            return 1;
        }
    }

    std::cout << "test_generated_spd passed\n";
    return 0;
}
