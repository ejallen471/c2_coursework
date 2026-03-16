/**
 * @file test_identity.cpp
 * @brief Checks that the identity matrix is preserved by factorisation.
 */

#include "mphil_dis_cholesky.h"
#include "test_helpers.h"

#include <iostream>
#include <vector>

int main()
{
    const int n = 5;
    std::vector<double> c = make_identity_matrix(n);
    const std::vector<double> expected = make_identity_matrix(n);

    const double elapsed = timed_cholesky_factorisation(c.data(), n);
    if (elapsed < 0.0)
    {
        std::cerr << "test_identity failed: routine returned " << elapsed << '\n';
        return 1;
    }

    if (!vectors_close(c, expected, 1e-12, 1e-12))
    {
        std::cerr << "test_identity failed: factorised matrix differs from identity\n";
        return 1;
    }

    std::cout << "test_identity passed\n";
    return 0;
}
