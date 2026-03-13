# How To Run

1. Build the project locally:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
```

2. Run one Cholesky version on one matrix:

```bash
./build/performance_tests/perf_time baseline 512
./build/performance_tests/perf_time lower_triangle_only 512
./build/performance_tests/perf_time inline_mirror 512
./build/performance_tests/perf_time loop_cleanup 512
./build/performance_tests/perf_time access_pattern 512
./build/performance_tests/perf_time cache_blocked 512
./build/performance_tests/perf_time vectorisation 512
./build/performance_tests/perf_time blocked_vectorised 512
```

3. Format for the command is:

```bash
./build/performance_tests/perf_time <optimisation> <matrix_size>
```

4. Run the tests:

```bash
ctest --test-dir build --output-on-failure
```

5. If comparing methods at one matrix size, use `perf_fixed_size_comparison` so the matrix is generated once and reused across methods:

```bash
./build/performance_tests/perf_fixed_size_comparison 1000 3 results/raw/fixed_size_comparison_n1000_raw.csv
```

6. If using the existing shell scripts instead, they build into `build`:

```bash
bash scripts/sh_scripts/build.sh
bash scripts/sh_scripts/run_single_matrix.sh baseline 5000
bash scripts/sh_scripts/run_fixed_size_comparison.sh 2000 1
bash scripts/sh_scripts/run_matrix_size_graph.sh baseline 5 64 128 256 512
bash scripts/sh_scripts/run_all_methods_graph.sh 3 2000 4000 6000
bash scripts/sh_scripts/run_block_size_sweep.sh 1000 5 16 24 32 48 64
```
