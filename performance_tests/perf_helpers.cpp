/**
 * @file perf_helpers.cpp
 * @brief Implementations of benchmark-reporting and validation helpers.
 */

#include "perf_helpers.h"

#include <algorithm>
#include <cmath>

#if defined(MPHIL_HAVE_LAPACK) && MPHIL_HAVE_LAPACK
extern "C"
{
    void dpotrf_(const char* uplo, const int* n, double* a, const int* lda,
                 int* info); // computes the Cholesky factorisation of a SPD matrix
}
#endif

// Function will compute the log-determinant from a matrix that has been factorised by Cholesky
LogDetValue logdet_from_factorised_storage(const std::vector<double>& c, std::size_t n)
{
    /*
    Formula is log det (A) = 2 sum of log(L_{ii})
    */

    LogDetValue sum = 0.0L;

    for (std::size_t i = 0; i < n; ++i)
    {
        const std::size_t index = i * n + i;
        sum += std::log(static_cast<LogDetValue>(c[index]));
    }

    return 2.0L * sum;
}

// Compute the relative error between the value and reference as a percentage
LogDetValue relative_difference_percent(LogDetValue value, LogDetValue reference)
{
    const LogDetValue scale = std::fabs(reference); // scale is the absolute size of the reference

    // If the reference is zero, return 100 percent
    if (scale == 0.0)
    {
        return (std::fabs(value) == 0.0)
            ? 0.0L
            : 100.0L; // condition: (std::fabs(value) == 0.0), if true return 0.0 else false 100.0
    }

    return 100.0L * std::fabs(value - reference) / scale;
}

std::vector<double> reconstruct_from_factorised_storage(const std::vector<double>& c, std::size_t n)
{
    std::vector<double> a(n * n, 0.0);

    for (std::size_t i = 0; i < n; ++i)
    {
        for (std::size_t j = 0; j < n; ++j)
        {
            LogDetValue sum = 0.0L;
            const std::size_t kmax = std::min(i, j);

            for (std::size_t k = 0; k <= kmax; ++k)
            {
                sum += static_cast<LogDetValue>(c[i * n + k]) *
                       static_cast<LogDetValue>(c[j * n + k]);
            }

            a[i * n + j] = static_cast<double>(sum);
        }
    }

    return a;
}

double relative_frobenius_error(const std::vector<double>& actual,
                                const std::vector<double>& reference)
{
    if (actual.size() != reference.size() || actual.empty())
    {
        return std::numeric_limits<double>::infinity();
    }

    long double numerator_sum = 0.0L;
    long double denominator_sum = 0.0L;

    for (std::size_t i = 0; i < actual.size(); ++i)
    {
        const long double diff =
            static_cast<long double>(actual[i]) - static_cast<long double>(reference[i]);
        numerator_sum += diff * diff;

        const long double ref = static_cast<long double>(reference[i]);
        denominator_sum += ref * ref;
    }

    if (denominator_sum == 0.0L)
    {
        return (numerator_sum == 0.0L) ? 0.0 : std::numeric_limits<double>::infinity();
    }

    return static_cast<double>(std::sqrt(numerator_sum / denominator_sum));
}

// Compute the log-determinant using LAPACK - we use this as the reference value
bool lapack_reference_logdet(std::vector<double> c, int n, LogDetValue& logdet)
{
#if defined(MPHIL_HAVE_LAPACK) && MPHIL_HAVE_LAPACK
    const char uplo = 'L'; // means use lower triangle
    const int lda = n;     // the leading dimension
    int info = 0;          // report sucess or failure

    // Call the LAPACK Cholesky factorisation on the matrix data to compute the logdet
    dpotrf_(&uplo, &n, c.data(), &lda, &info);

    // If failed, return false
    if (info != 0)
    {
        return false;
    }

    // if factorisation was sucessful, return true
    logdet = logdet_from_factorised_storage(c, n);
    return true;
#else
    (void)c;
    (void)n;
    (void)logdet;
    return false;
#endif
}

std::string quoted_path(const std::filesystem::path& path)
{
    // Quote paths before passing them to std::system so spaces do not break the command line.
    return "\"" + path.string() + "\"";
}
