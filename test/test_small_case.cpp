#include "mphil_dis_cholesky.h"
#include "test_helpers.h"

#include <cmath>
#include <iostream>
#include <string>
#include <vector>

bool run_case(const std::string& name, std::vector<double> c, const std::vector<double>& expected,
              int n)
{
    const double elapsed = mphil_dis_cholesky(c.data(), n);

    if (elapsed < 0.0)
    {
        std::cerr << "test_small_case failed for " << name << ": routine returned " << elapsed
                  << '\n';
        return false;
    }

    for (int i = 0; i < n * n; ++i)
    {
        if (!nearly_equal(c[i], expected[i], 1e-12, 1e-12))
        {
            std::cerr << "test_small_case failed for " << name << " at index " << i << ": got "
                      << c[i] << ", expected " << expected[i] << '\n';
            return false;
        }
    }

    return true;
}

int main()
{
    {
        const int n = 2;
        const std::vector<double> c = {4.0, 2.0, 2.0, 26.0};
        const std::vector<double> expected = {2.0, 1.0, 1.0, 5.0};

        if (!run_case("brief 2x2 example", c, expected, n))
        {
            return 1;
        }
    }

    {
        const int n = 2;
        const std::vector<double> c = {9.0, 3.0, 3.0, 5.0};
        const std::vector<double> expected = {3.0, 1.0, 1.0, 2.0};

        if (!run_case("2x2 SPD case", c, expected, n))
        {
            return 1;
        }
    }

    {
        const int n = 3;
        const std::vector<double> c = {4.0, 2.0, 2.0, 2.0, 10.0, 5.0, 2.0, 5.0, 9.0};
        const std::vector<double> expected = {
            2.0, 1.0, 1.0, 1.0, 3.0, 4.0 / 3.0, 1.0, 4.0 / 3.0, std::sqrt(56.0 / 9.0)};

        if (!run_case("3x3 SPD case", c, expected, n))
        {
            return 1;
        }
    }

    std::cout << "test_small_case passed\n";
    return 0;
}