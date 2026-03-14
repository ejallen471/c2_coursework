# Block-Size Sweep

This workflow sweeps the block size for one blocked optimisation and produces a CSV plus plots.

Build first:

```bash
bash scripts/sh_scripts/build.sh
```

Run the plotting workflow:

```bash
bash scripts/sh_scripts/run_block_size_sweep.sh cache_blocked_1 1000 5 16 24 32 48 64
```

Direct benchmark command without plotting:

```bash
./build/run/run_cholesky block-size-sweep cache_blocked_1 1000 5 results/raw/cache_blocked_block_size_sweep_n1000.csv 16 24 32 48 64
```

Arguments:

```text
./build/run/run_cholesky block-size-sweep <optimisation> <matrix_size> <repeats> <raw_csv> <block_size1> [block_size2 ...]
```

Blocked optimisation names:

```text
cache_blocked_1
cache_blocked_2
```

Expected Outputs:

c2_coursework/results/raw/cache_blocked_1_block_size_sweep_n1000.csv

c2_coursework/results/figures/cache_blocked_1_block_size_sweep_n1000/summary_by_block_size.csv
c2_coursework/results/figures/cache_blocked_1_block_size_sweep_n1000/runtime_vs_block_size.png
c2_coursework/results/figures/cache_blocked_1_block_size_sweep_n1000/speedup_vs_block_size.png
