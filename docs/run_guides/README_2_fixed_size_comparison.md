# Fixed-Size Comparison

This workflow runs the selected single-threaded optimisations on one matrix size and writes a CSV.

Build first:

```bash
bash scripts/sh_scripts/build.sh
```

Run directly:

```bash
./build/run/run_cholesky fixed-size 2000 3 results/raw/fixed_size_comparison_n2000_raw.csv
```

Run a subset of methods:

```bash
./build/run/run_cholesky fixed-size 20000 3 results/raw/fixed_size_comparison_n20000_raw.csv \
  lower_triangle upper_triangle contiguous_access cache_blocked_1 cache_blocked_2
```

Arguments:

```text
./build/run/run_cholesky fixed-size <matrix_size> <repeats> <raw_csv> [method1 method2 ...]
```

Output:

- raw CSV with one row per optimisation per repeat
- speedup versus baseline when `baseline` is included, otherwise `speedup_factor_vs_baseline` is `nan`
- log-determinant comparison fields
