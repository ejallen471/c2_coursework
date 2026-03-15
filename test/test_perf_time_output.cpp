#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>

namespace
{
std::map<std::string, std::string> parse_fields(const std::string& line)
{
    std::map<std::string, std::string> fields;
    std::istringstream stream(line);
    std::string token;

    while (stream >> token)
    {
        const std::size_t pos = token.find('=');
        if (pos == std::string::npos)
        {
            continue;
        }

        fields[token.substr(0, pos)] = token.substr(pos + 1);
    }

    return fields;
}

bool parse_double_field(const std::map<std::string, std::string>& fields, const std::string& key,
                        double& value)
{
    const auto it = fields.find(key);
    if (it == fields.end())
    {
        return false;
    }

    char* end = nullptr;
    value = std::strtod(it->second.c_str(), &end);
    return end != nullptr && *end == '\0';
}
} // namespace

int main(int argc, char* argv[])
{
    if (argc != 2)
    {
        std::cerr << "Usage: " << argv[0] << " <run_cholesky_executable>\n";
        return 1;
    }

    const std::filesystem::path csv_path =
        std::filesystem::temp_directory_path() / "test_perf_time_output.csv";
    std::filesystem::remove(csv_path);

    const std::string command =
        "\"" + std::string(argv[1]) + "\" time baseline 16 \"" + csv_path.string() + "\"";
    FILE* pipe = popen(command.c_str(), "r");
    if (pipe == nullptr)
    {
        std::cerr << "test_perf_time_output failed: popen failed\n";
        return 1;
    }

    std::string output;
    char buffer[256];
    while (std::fgets(buffer, sizeof(buffer), pipe) != nullptr)
    {
        output += buffer;
    }

    const int status = pclose(pipe);
    if (status != 0)
    {
        std::cerr << "test_perf_time_output failed: run_cholesky exited with status " << status
                  << '\n';
        return 1;
    }

    const std::map<std::string, std::string> fields = parse_fields(output);

    if (fields.count("optimisation") == 0 || fields.at("optimisation") != "baseline")
    {
        std::cerr << "test_perf_time_output failed: missing or wrong optimisation field\n";
        return 1;
    }

    if (fields.count("n") == 0 || fields.at("n") != "16")
    {
        std::cerr << "test_perf_time_output failed: missing or wrong n field\n";
        return 1;
    }

    if (fields.count("raw_csv") == 0 || fields.at("raw_csv") != "\"" + csv_path.string() + "\"")
    {
        std::cerr << "test_perf_time_output failed: missing or wrong raw_csv field\n";
        return 1;
    }

    double elapsed = 0.0;
    double logdet_factor = 0.0;
    if (!parse_double_field(fields, "elapsed_seconds", elapsed) || elapsed < 0.0)
    {
        std::cerr << "test_perf_time_output failed: missing or invalid elapsed_seconds field\n";
        return 1;
    }

    if (!parse_double_field(fields, "logdet_factor", logdet_factor))
    {
        std::cerr << "test_perf_time_output failed: missing or invalid logdet_factor field\n";
        return 1;
    }

    const auto library_it = fields.find("logdet_library");
    const auto diff_it = fields.find("relative_difference_percent");
    if (library_it == fields.end() || diff_it == fields.end())
    {
        std::cerr << "test_perf_time_output failed: missing comparison fields\n";
        return 1;
    }

    if (library_it->second != "unavailable")
    {
        double logdet_library = 0.0;
        double relative_difference = 0.0;

        if (!parse_double_field(fields, "logdet_library", logdet_library) ||
            !parse_double_field(fields, "relative_difference_percent", relative_difference))
        {
            std::cerr << "test_perf_time_output failed: comparison fields are not numeric\n";
            return 1;
        }

        if (relative_difference > 1.0e-8)
        {
            std::cerr << "test_perf_time_output failed: relative difference is too large ("
                      << relative_difference << "%)\n";
            return 1;
        }
    }
    else if (diff_it->second != "unavailable")
    {
        std::cerr << "test_perf_time_output failed: unavailable reference should also report unavailable difference\n";
        return 1;
    }

    std::ifstream csv_file(csv_path);
    if (!csv_file)
    {
        std::cerr << "test_perf_time_output failed: CSV output file was not created\n";
        return 1;
    }

    std::string header;
    std::string row;
    if (!std::getline(csv_file, header) || !std::getline(csv_file, row))
    {
        std::cerr << "test_perf_time_output failed: CSV output did not contain two lines\n";
        return 1;
    }

    if (header != "optimisation,n,elapsed_seconds,logdet_library,logdet_factor,relative_difference_percent")
    {
        std::cerr << "test_perf_time_output failed: unexpected CSV header\n";
        return 1;
    }

    if (row.rfind("baseline,16,", 0) != 0)
    {
        std::cerr << "test_perf_time_output failed: unexpected CSV data row\n";
        return 1;
    }

    std::filesystem::remove(csv_path);

    std::cout << "test_perf_time_output passed\n";
    return 0;
}
