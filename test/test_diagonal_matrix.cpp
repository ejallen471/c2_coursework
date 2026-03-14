#include "mphil_dis_cholesky.h"
#include "test_helpers.h"

#include <iostream>
#include <vector>

int main()
{
    const std::vector<double> diag = {4.0, 9.0, 16.0, 25.0};
    const int n = static_cast<int>(diag.size());

    std::vector<double> c = make_diagonal_matrix(diag);

    const double elapsed = timed_cholesky_factorisation(c.data(), n);
    if (elapsed < 0.0)
    {
        std::cerr << "test_diagonal_matrix failed: routine returned " << elapsed << '\n';
        return 1;
    }

    const std::vector<double> expected = {2.0, 0.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0,
                                          0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 0.0, 5.0};

    if (!vectors_close(c, expected, 1e-12, 1e-12))
    {
        std::cerr << "test_diagonal_matrix failed\n";
        return 1;
    }

    std::cout << "test_diagonal_matrix passed\n";
    return 0;
}
