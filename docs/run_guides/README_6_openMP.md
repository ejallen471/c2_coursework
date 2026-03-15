# OpenMP Workflows

These workflows use the OpenMP-enabled build in `build_openmp/`.

Build first:

```bash
bash scripts/sh_scripts/build_openmp.sh
```

For plotting locally, make sure your Python environment has:

```bash
pip install -r requirements-plot.txt
```

## 1. Fixed-Size Comparison

This workflow runs:

- `baseline` with `OMP_NUM_THREADS=1`
- `openmp1`
- `openmp2`
- `openmp3`
- `openmp4`

at one matrix size, then writes one combined CSV.

Run:

```bash
OPENMP_THREADS=8 bash scripts/sh_scripts/run_openmp_fixed_size_comparison.sh 2000 1
```

Arguments:

```text
bash scripts/sh_scripts/run_openmp_fixed_size_comparison.sh <matrix_size> <repeats> [raw_csv]
```

Expected output:

- `results/raw/openmp_fixed_size_comparison_n2000.csv`
  One row per method and repeat, with:
  `optimisation,n,repeat,elapsed_seconds,speedup_factor_vs_baseline,logdet_library,logdet_factor,relative_difference_percent`

Notes:

- `OPENMP_THREADS` controls the thread count for `openmp1` to `openmp4`
- `baseline` is always run with one thread for a fair single-thread reference

## 2. Matrix-Size Scaling

This workflow runs:

- `baseline` with `OMP_NUM_THREADS=1`
- `openmp1`
- `openmp2`
- `openmp3`
- `openmp4`

across several matrix sizes and generates combined comparison plots.

Run:

```bash
OPENMP_THREADS=8 bash scripts/sh_scripts/run_openmp_methods_graph.sh 3 512 1024 2000 4096
```

Arguments:

```text
bash scripts/sh_scripts/run_openmp_methods_graph.sh <repeats> <n1> [n2 ...]
```

Expected outputs:

- `results/raw/openmp_methods_graph.csv`
  Raw timing data for all requested methods, matrix sizes, and repeats.
- `results/figures/openmp_methods_graph/runtime_vs_n_by_method.png`
  Log-log runtime plot comparing `baseline` and all OpenMP methods.
- `results/figures/openmp_methods_graph/speedup_vs_baseline.png`
  Speedup relative to the single-thread baseline as matrix size changes.
- `results/figures/openmp_methods_graph/cubic_scaling_check_by_method.png`
  Normalised log-log scaling plot with an `n^3` reference line.

Notes:

- the plots include error bars when repeats are greater than 1
- `OPENMP_THREADS` controls the thread count for the OpenMP methods only
- `baseline` is kept single-threaded throughout

## 3. Thread-Count Sweep

This workflow measures how the OpenMP methods scale with thread count at one fixed matrix size.

Run:

```bash
bash scripts/sh_scripts/run_openmp_thread_count_sweep.sh 2000 3 1 2 4 8
```

Arguments:

```text
bash scripts/sh_scripts/run_openmp_thread_count_sweep.sh <matrix_size> <repeats> <threads1> [threads2 ...]
```

Expected outputs:

- `results/raw/openmp_thread_count_sweep_n2000.csv`
  Raw timing data for `baseline` and all OpenMP methods across the requested thread counts.
- `results/figures/openmp_thread_count_sweep_n2000/summary_by_threads.csv`
  Aggregated summary by method and thread count.
- `results/figures/openmp_thread_count_sweep_n2000/speedup_vs_thread_count.png`
  Speedup relative to each method's own 1-thread performance.

Notes:

- `baseline` is included as a 1-thread reference row in the raw CSV
- the plot focuses on `openmp1` to `openmp4`
- the plot includes error bars when repeats are greater than 1
