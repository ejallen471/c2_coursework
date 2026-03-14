# Fixed-Size Comparison

This workflow runs all implemented single-threaded optimisations on one matrix size and writes a CSV.

Build first:

```bash
bash scripts/sh_scripts/build.sh
```

Run directly:

```bash
./build/run/run_cholesky fixed-size 2000 3 results/raw/fixed_size_comparison_n2000_raw.csv
```

Arguments:

```text
./build/run/run_cholesky fixed-size <matrix_size> <repeats> <raw_csv>
```

Output:

- raw CSV with one row per optimisation per repeat
- speedup versus baseline
- log-determinant comparison fields
