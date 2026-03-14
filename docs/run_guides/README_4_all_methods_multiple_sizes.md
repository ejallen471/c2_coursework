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

- one raw CSV per method in `results/raw/all_methods_graph/`
- one plot directory per method in `results/figures/all_methods_graph/`
- one combined comparison plot directory in `results/figures/all_methods_graph/combined/`
