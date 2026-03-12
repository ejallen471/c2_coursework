#include "mphil_dis_cholesky.h"
#include "test_helpers.h"

#include <iostream>
#include <vector>

int main()
{
    const int n = 8;
    const std::vector<double> original = make_generated_spd_matrix(n);
    std::vector<double> c = original;

    const double elapsed = mphil_dis_cholesky(c.data(), n);
    if (elapsed < 0.0)
    {
        std::cerr << "test_reconstruction failed: routine returned " << elapsed << '\n';
        return 1;
    }

    const std::vector<double> reconstructed = reconstruct_from_factorised_storage(c, n);

    if (!vectors_close(reconstructed, original, 1e-10, 1e-10))
    {
        std::cerr << "test_reconstruction failed: reconstructed matrix does not match original\n";
        return 1;
    }

    std::cout << "test_reconstruction passed\n";
    return 0;
}