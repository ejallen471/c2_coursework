#include "runtime_cholesky.h"

#include <iostream>
#include <string>
#include <utility>
#include <vector>

int main()
{
    const std::vector<std::pair<std::string, CholeskyVersion>> valid_cases = {
        {"baseline", CholeskyVersion::Baseline},
        {"lower_triangle", CholeskyVersion::LowerTriangleOnly},
        {"lowertriangle", CholeskyVersion::LowerTriangleOnly},
        {"lower_triangle_only", CholeskyVersion::LowerTriangleOnly},
        {"upper_triangle", CholeskyVersion::UpperTriangle},
        {"uppertriangle", CholeskyVersion::UpperTriangle},
        {"triangular_upper", CholeskyVersion::UpperTriangle},
        {"contiguous_access", CholeskyVersion::ContiguousAccess},
        {"contiguousaccess", CholeskyVersion::ContiguousAccess},
        {"access_pattern", CholeskyVersion::ContiguousAccess},
        {"cache_blocked_1", CholeskyVersion::cacheBlockedOne},
        {"cacheblocked1", CholeskyVersion::cacheBlockedOne},
        {"cache_blocked_2", CholeskyVersion::cacheBlockedTwo},
        {"cacheblocked2", CholeskyVersion::cacheBlockedTwo},
        {"openmp1", CholeskyVersion::OpenMP1},
        {"openmp2", CholeskyVersion::OpenMP2},
        {"openmp3", CholeskyVersion::OpenMP3},
        {"openmp4", CholeskyVersion::OpenMP4},
    };

    for (const auto& valid_case : valid_cases)
    {
        CholeskyVersion parsed = CholeskyVersion::Baseline;
        if (!parse_optimisation_name(valid_case.first, parsed))
        {
            std::cerr << "test_runtime_name_parsing failed: could not parse '"
                      << valid_case.first << "'\n";
            return 1;
        }

        if (parsed != valid_case.second)
        {
            std::cerr << "test_runtime_name_parsing failed: parsed '" << valid_case.first
                      << "' to the wrong version\n";
            return 1;
        }
    }

    const CholeskyVersion canonical_versions[] = {
        CholeskyVersion::Baseline,          CholeskyVersion::LowerTriangleOnly,
        CholeskyVersion::UpperTriangle,     CholeskyVersion::ContiguousAccess,
        CholeskyVersion::cacheBlockedOne,   CholeskyVersion::cacheBlockedTwo,
        CholeskyVersion::OpenMP1,           CholeskyVersion::OpenMP2,
        CholeskyVersion::OpenMP3,           CholeskyVersion::OpenMP4,
    };

    for (const CholeskyVersion version : canonical_versions)
    {
        CholeskyVersion parsed = CholeskyVersion::Baseline;
        const char* name = optimisation_name(version);
        if (!parse_optimisation_name(name, parsed))
        {
            std::cerr << "test_runtime_name_parsing failed: could not round-trip '" << name
                      << "'\n";
            return 1;
        }

        if (parsed != version)
        {
            std::cerr << "test_runtime_name_parsing failed: wrong round-trip version for '"
                      << name << "'\n";
            return 1;
        }
    }

    CholeskyVersion parsed = CholeskyVersion::Baseline;
    if (parse_optimisation_name("openmp5", parsed))
    {
        std::cerr << "test_runtime_name_parsing failed: unexpected parse success for 'openmp5'\n";
        return 1;
    }

    std::cout << "test_runtime_name_parsing passed\n";
    return 0;
}
