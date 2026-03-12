#ifndef TEST_HELPERS_H
#define TEST_HELPERS_H

#include "matrix.h"

#include <vector>

bool nearly_equal(double a, double b, double atol = 1e-12, double rtol = 1e-12);

bool vectors_close(const std::vector<double>& a, const std::vector<double>& b, double atol = 1e-12,
                   double rtol = 1e-12);

bool matrix_is_symmetric(const std::vector<double>& a, int n, double atol = 1e-12,
                         double rtol = 1e-12);

bool diagonal_is_positive(const std::vector<double>& a, int n);

void print_matrix(const std::vector<double>& a, int n);

std::vector<double> make_identity_matrix(int n);
std::vector<double> make_diagonal_matrix(const std::vector<double>& diag);
std::vector<double> make_brief_example_matrix();

std::vector<double> lower_factor_from_storage(const std::vector<double>& c, int n);
std::vector<double> reconstruct_from_factorised_storage(const std::vector<double>& c, int n);

double logdet_from_factorised_storage(const std::vector<double>& c, int n);

#endif
