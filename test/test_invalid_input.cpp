/**
 * @file test_invalid_input.cpp
 * @brief Exercises the public timing entry point with invalid inputs.
 */

#include "mphil_dis_cholesky.h"

#include <iostream>

int main()
{
    {
        const double result = timed_cholesky_factorisation(nullptr, 2);
        if (result >= 0.0)
        {
            std::cerr << "test_invalid_input failed: nullptr input did not fail\n";
            return 1;
        }
    }

    {
        double c[4] = {1.0, 0.0, 0.0, 1.0};
        const double result = timed_cholesky_factorisation(c, 0);
        if (result >= 0.0)
        {
            std::cerr << "test_invalid_input failed: n = 0 did not fail\n";
            return 1;
        }
    }

    {
        double c[1] = {1.0};
        const double result = timed_cholesky_factorisation(c, 100001);
        if (result >= 0.0)
        {
            std::cerr << "test_invalid_input failed: n > 100000 did not fail\n";
            return 1;
        }
    }

    std::cout << "test_invalid_input passed\n";
    return 0;
}
