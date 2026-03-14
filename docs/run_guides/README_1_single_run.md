# Single Run

This workflow runs one optimisation for one matrix size.

Build first:

```bash
bash scripts/sh_scripts/build.sh
```

Run without CSV:

```bash
./build/run/run_cholesky time baseline 512
```

Run and write one-row CSV (Choose one of the following):

```bash
./build/run/run_cholesky time baseline 1000 results/raw/single_run_baseline_n512.csv
./build/run/run_cholesky time lower_triangle 1000 results/raw/single_run_lower_triangle_n512.csv
./build/run/run_cholesky time upper_triangle 1000 results/raw/single_run_upper_triangle_n512.csv
./build/run/run_cholesky time contiguous_access 1000 results/raw/single_run_contiguous_access_n512.csv
./build/run/run_cholesky time cache_blocked_1 1000 results/raw/single_run_cache_blocked_1_n512.csv
./build/run/run_cholesky time cache_blocked_2 1000 results/raw/single_run_cache_blocked_2_n512.csv
```

Arguments:

```text
./build/run/run_cholesky time <optimisation> <matrix_size> [raw_csv_file_name]
```

Optimisation Choices:

```text
baseline
lower_triangle
upper_triangle
contiguous_access
cache_blocked_1
cache_blocked_2
```
