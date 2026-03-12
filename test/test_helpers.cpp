#include "test_helpers.h"

#include <algorithm>
#include <cmath>
#include <iostream>

bool nearly_equal(double a, double b, double atol, double rtol)
{
    const double diff = std::fabs(a - b);
    const double scale = std::max(std::fabs(a), std::fabs(b));
    return diff <= atol + rtol * scale;
}

bool vectors_close(const std::vector<double>& a, const std::vector<double>& b, double atol,
                   double rtol)
{
    if (a.size() != b.size())
    {
        return false;
    }

    for (std::size_t i = 0; i < a.size(); ++i)
    {
        if (!nearly_equal(a[i], b[i], atol, rtol))
        {
            return false;
        }
    }

    return true;
}

bool matrix_is_symmetric(const std::vector<double>& a, int n, double atol, double rtol)
{
    if (a.size() != static_cast<std::size_t>(n) * static_cast<std::size_t>(n))
    {
        return false;
    }

    for (int i = 0; i < n; ++i)
    {
        for (int j = i + 1; j < n; ++j)
        {
            if (!nearly_equal(a[static_cast<std::size_t>(i) * static_cast<std::size_t>(n) +
                                    static_cast<std::size_t>(j)],
                              a[static_cast<std::size_t>(j) * static_cast<std::size_t>(n) +
                                    static_cast<std::size_t>(i)],
                              atol,
                              rtol))
            {
                return false;
            }
        }
    }

    return true;
}

bool diagonal_is_positive(const std::vector<double>& a, int n)
{
    if (a.size() != static_cast<std::size_t>(n) * static_cast<std::size_t>(n))
    {
        return false;
    }

    for (int i = 0; i < n; ++i)
    {
        if (a[static_cast<std::size_t>(i) * static_cast<std::size_t>(n) +
              static_cast<std::size_t>(i)] <= 0.0)
        {
            return false;
        }
    }

    return true;
}

void print_matrix(const std::vector<double>& a, int n)
{
    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            std::cout << a[i * n + j] << ' ';
        }
        std::cout << '\n';
    }
}

std::vector<double> make_identity_matrix(int n)
{
    std::vector<double> a(n * n, 0.0);

    for (int i = 0; i < n; ++i)
    {
        a[i * n + i] = 1.0;
    }

    return a;
}

std::vector<double> make_diagonal_matrix(const std::vector<double>& diag)
{
    const int n = static_cast<int>(diag.size());
    std::vector<double> a(n * n, 0.0);

    for (int i = 0; i < n; ++i)
    {
        a[i * n + i] = diag[i];
    }

    return a;
}

std::vector<double> make_brief_example_matrix()
{
    return {4.0, 2.0, 2.0, 26.0};
}

std::vector<double> lower_factor_from_storage(const std::vector<double>& c, int n)
{
    std::vector<double> l(n * n, 0.0);

    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j <= i; ++j)
        {
            l[i * n + j] = c[i * n + j];
        }
    }

    return l;
}

std::vector<double> reconstruct_from_factorised_storage(const std::vector<double>& c, int n)
{
    const std::vector<double> l = lower_factor_from_storage(c, n);
    std::vector<double> a(n * n, 0.0);

    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            double sum = 0.0;
            const int kmax = std::min(i, j);

            for (int k = 0; k <= kmax; ++k)
            {
                sum += l[i * n + k] * l[j * n + k];
            }

            a[i * n + j] = sum;
        }
    }

    return a;
}

double logdet_from_factorised_storage(const std::vector<double>& c, int n)
{
    double sum = 0.0;

    for (int i = 0; i < n; ++i)
    {
        sum += std::log(c[i * n + i]);
    }

    return 2.0 * sum;
}
