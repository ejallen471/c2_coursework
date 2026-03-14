#ifndef PERF_HELPERS_H
#define PERF_HELPERS_H

#include <cstddef>
#include <filesystem>
#include <limits>
#include <string>
#include <vector>

using LogDetValue = long double;

inline constexpr int kLogDetOutputPrecision = std::numeric_limits<LogDetValue>::max_digits10;

LogDetValue logdet_from_factorised_storage(const std::vector<double>& c, std::size_t n);
LogDetValue relative_difference_percent(LogDetValue value, LogDetValue reference);
bool lapack_reference_logdet(std::vector<double> c, int n, LogDetValue& logdet);
std::string quoted_path(const std::filesystem::path& path);

#endif
