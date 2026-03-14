#include "mphil_dis_cholesky.h"
#include "test_helpers.h"

#include <iostream>
#include <vector>

int main()
{
    const int n = 10;
    const std::vector<double> original = make_generated_spd_matrix(n);

    std::vector<double> reference = original;
    const double first_elapsed = timed_cholesky_factorisation(reference.data(), n);
    if (first_elapsed < 0.0)
    {
        std::cerr << "test_repeatability failed on first run: routine returned " << first_elapsed
                  << '\n';
        return 1;
    }

    for (int run = 0; run < 5; ++run)
    {
        std::vector<double> current = original;
        const double elapsed = timed_cholesky_factorisation(current.data(), n);

        if (elapsed < 0.0)
        {
            std::cerr << "test_repeatability failed on run " << run << ": routine returned "
                      << elapsed << '\n';
            return 1;
        }

        if (!vectors_close(current, reference, 1e-12, 1e-12))
        {
            std::cerr << "test_repeatability failed on run " << run
                      << ": factorised output differs from reference run\n";
            return 1;
        }
    }

    std::cout << "test_repeatability passed\n";
    return 0;
}
