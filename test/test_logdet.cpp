#include "mphil_dis_cholesky.h"
#include "test_helpers.h"

#include <cmath>
#include <iostream>
#include <vector>

int main()
{
    {
        const int n = 2;
        std::vector<double> c = make_brief_example_matrix();

        const double elapsed = timed_cholesky_factorisation(c.data(), n);
        if (elapsed < 0.0)
        {
            std::cerr << "test_logdet failed on 2x2 example: routine returned " << elapsed << '\n';
            return 1;
        }

        const double got = logdet_from_factorised_storage(c, n);
        const double expected = std::log(100.0);

        if (!nearly_equal(got, expected, 1e-12, 1e-12))
        {
            std::cerr << "test_logdet failed on 2x2 example: got " << got << ", expected "
                      << expected << '\n';
            return 1;
        }
    }

    {
        const std::vector<double> diag = {4.0, 9.0, 16.0};
        const int n = static_cast<int>(diag.size());
        std::vector<double> c = make_diagonal_matrix(diag);

        const double elapsed = timed_cholesky_factorisation(c.data(), n);
        if (elapsed < 0.0)
        {
            std::cerr << "test_logdet failed on diagonal example: routine returned " << elapsed
                      << '\n';
            return 1;
        }

        const double got = logdet_from_factorised_storage(c, n);
        const double expected = std::log(4.0 * 9.0 * 16.0);

        if (!nearly_equal(got, expected, 1e-12, 1e-12))
        {
            std::cerr << "test_logdet failed on diagonal example: got " << got << ", expected "
                      << expected << '\n';
            return 1;
        }
    }

    std::cout << "test_logdet passed\n";
    return 0;
}
