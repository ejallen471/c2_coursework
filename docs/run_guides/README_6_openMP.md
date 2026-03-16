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

This workflow runs the selected OpenMP fixed-size methods at one matrix size, then writes one combined CSV.

Run:

```bash
OPENMP_THREADS=8 bash scripts/sh_scripts/run_openmp_fixed_size_comparison.sh 2000 1
```

Run a subset of methods:

```bash
OPENMP_THREADS=32 bash scripts/sh_scripts/run_openmp_fixed_size_comparison.sh \
  20000 \
  3 \
  results/raw/openmp_fixed_size_comparison_n20000.csv \
  --methods openmp1 openmp2 openmp3 openmp4
```

Arguments:

```text
bash scripts/sh_scripts/run_openmp_fixed_size_comparison.sh <matrix_size> <repeats> [raw_csv] [--methods <method1> [method2 ...]]
```

Expected output:

- `results/raw/openmp_fixed_size_comparison_n2000.csv`
  One row per method and repeat, with:
  `optimisation,n,repeat,elapsed_seconds,speedup_factor_vs_baseline,logdet_library,logdet_factor,relative_difference_percent`

Notes:

- `OPENMP_THREADS` controls the thread count for `openmp1` to `openmp4`
- valid method names are `baseline`, `openmp1`, `openmp2`, `openmp3`, and `openmp4`
- `baseline` is always run with one thread when it is included
- if `baseline` is omitted, the CSV is still written but `speedup_factor_vs_baseline` is `nan`

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

Run a selected subset of methods:

```bash
OPENMP_THREADS=8 bash scripts/sh_scripts/run_openmp_methods_graph.sh \
  3 \
  --methods openmp1 openmp2 openmp3 openmp4 \
  --sizes 256 512 1000 2000 4000 8000 12000 16000 20000
```

Arguments:

```text
bash scripts/sh_scripts/run_openmp_methods_graph.sh <repeats> <n1> [n2 ...]
bash scripts/sh_scripts/run_openmp_methods_graph.sh <repeats> --methods <method1> [method2 ...] --sizes <n1> [n2 ...]
```

Expected outputs:

- `results/raw/openmp_methods_graph.csv`
  Raw timing data for all requested methods, matrix sizes, and repeats.
- `results/figures/openmp_methods_graph/runtime_vs_n_by_method.png`
  Log-log runtime plot comparing `baseline` and all OpenMP methods.
- `results/figures/openmp_methods_graph/speedup_vs_baseline.png`
  Speedup relative to the single-thread baseline as matrix size changes, when `baseline` is included.
- `results/figures/openmp_methods_graph/cubic_scaling_check_by_method.png`
  Normalised log-log scaling plot with an `n^3` reference line.

Notes:

- the plots include error bars when repeats are greater than 1
- `OPENMP_THREADS` controls the thread count for the OpenMP methods only
- `baseline` is kept single-threaded throughout
- if `baseline` is omitted from `--methods`, the runtime and cubic-scaling plots are still produced, but the speedup plot is skipped

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
  Raw timing data for `openmp1` to `openmp4` across the requested thread counts.
- `results/figures/openmp_thread_count_sweep_n2000/summary_by_threads.csv`
  Aggregated summary by method and thread count.
- `results/figures/openmp_thread_count_sweep_n2000/speedup_vs_thread_count.png`
  Speedup relative to each method's own 1-thread performance.

Notes:

- the plot focuses on `openmp1` to `openmp4`
- include a `1`-thread run if you want the speedup plot, because speedup is measured relative to each method's own 1-thread timing
- the plot includes error bars when repeats are greater than 1
