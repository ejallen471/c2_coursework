# All Methods Across Multiple Sizes

This workflow runs all single-threaded methods across many matrix sizes and creates one combined comparison plot.

Build first:

```bash
bash scripts/sh_scripts/build.sh
```

Run:

```bash
bash scripts/sh_scripts/run_all_methods_graph.sh 3 2000 4000 6000
```

Outputs:

- one combined raw CSV in `results/raw/`
- one combined plot directory in `results/figures/all_methods_graph/`

Expected files:

- `results/raw/all_methods_graph.csv`
  One combined raw dataset containing all methods, all requested matrix sizes, and all repeats.
- `results/figures/all_methods_graph/runtime_vs_n_by_method.png`
  One comparison plot with all methods shown together as runtime-vs-size curves.
- `results/figures/all_methods_graph/speedup_vs_baseline.png`
  One comparison plot showing each method's speedup relative to baseline across matrix sizes.


Optimisation Choices:

```text
baseline
lower_triangle
upper_triangle
contiguous_access
cache_blocked_1
cache_blocked_2
```
