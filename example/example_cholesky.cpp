#include "matrix.h"
#include "mphil_dis_cholesky.h"

#include <iomanip>
#include <iostream>
#include <vector>

void print_matrix(const double* c, int n)
{
    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            std::cout << std::setw(10) << c[i * n + j] << ' ';
        }
        std::cout << '\n';
    }
}

int main()
{
    const int n = 4;

    MatrixGenerationOptions options;
    options.kernel = CovarianceKernel::SquaredExponential;
    options.randomize_points = false;

    std::vector<double> c = make_generated_spd_matrix(n, options);

    std::cout << "Input matrix:\n";
    print_matrix(c.data(), n);

    const double elapsed = mphil_dis_cholesky(c.data(), n);

    std::cout << "\nFactorised matrix:\n";
    print_matrix(c.data(), n);

    std::cout << "\nElapsed time: " << elapsed << " seconds\n";

    return 0;
}
