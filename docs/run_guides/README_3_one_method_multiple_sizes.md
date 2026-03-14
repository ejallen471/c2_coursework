# One Method Across Multiple Sizes

This workflow runs one optimisation across many matrix sizes and produces both a CSV and plots.

Build first:

```bash
bash scripts/sh_scripts/build.sh
```

Use the shell script:

```bash
bash scripts/sh_scripts/run_matrix_size_graph.sh baseline 3 100 1000 1500 2000
```

What it does:

- runs `run_cholesky scaling`
- writes a raw CSV to `results/raw/`
- generates plots in `results/figures/`

Expected outputs:

- `results/raw/<optimisation>_matrix_size_graph.csv`
  Raw timing data with one row per matrix size and repeat.
- `results/figures/<optimisation>_matrix_size_graph/summary_by_n.csv`
  Aggregated statistics by matrix size, including mean, median, standard deviation, and `T(n)/n^3`.
- `results/figures/<optimisation>_matrix_size_graph/runtime_vs_n.png`
  A log-log runtime scaling plot showing how execution time grows with matrix size.
- `results/figures/<optimisation>_matrix_size_graph/cubic_scaling_check.png`
  A normalised log-log scaling plot with an `n^3` reference line used to check whether the measured growth is approximately cubic.

Note:
- the plots include error bars when repeats are greater than 1

Note these graphs are meant to show long term behaviour therefore a wide range of plotting values is best

Direct benchmark command without plotting:

```bash
./build/run/run_cholesky scaling baseline 5 results/raw/baseline_matrix_size_graph.csv 3 100 1000 1500 2000
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
