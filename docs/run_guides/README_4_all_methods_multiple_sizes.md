# All Methods Across Multiple Sizes

This workflow runs all single-threaded methods across many matrix sizes and produces one merged raw CSV plus one combined plot directory.

Build first:

```bash
bash scripts/sh_scripts/build.sh
```

For plotting locally, make sure your Python environment has:

```bash
pip install -r requirements-plot.txt
```

Run all single-threaded methods:

```bash
bash scripts/sh_scripts/run_all_methods_graph.sh 3 2000 4000 6000
```

Run a selected subset of methods:

```bash
bash scripts/sh_scripts/run_all_methods_graph.sh \
  3 \
  --methods lower_triangle upper_triangle contiguous_access cache_blocked_1 cache_blocked_2 \
  --sizes 256 384 512 768 1000 1500 2000 3000 4000 6000 8000 12000 16000 20000
```

Outputs:

- `results/raw/all_methods_graph.csv`
  One merged raw CSV containing all requested methods, matrix sizes, and repeats.
- `results/figures/all_methods_graph/runtime_vs_n_by_method.png`
  Log-log runtime comparison across all single-thread methods.
- `results/figures/all_methods_graph/speedup_vs_baseline.png`
  Speedup relative to `baseline` as matrix size changes, when `baseline` is included.
- `results/figures/all_methods_graph/cubic_scaling_check_by_method.png`
  Normalised log-log scaling plot with an `n^3` reference line.

Notes:

- the plots include error bars when repeats are greater than 1
- the workflow creates temporary per-method CSVs internally, then merges them into the single final raw CSV
- if `baseline` is omitted from `--methods`, the runtime and cubic-scaling plots are still produced, but the speedup plot is skipped
