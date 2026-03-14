# Running Locally

Build the project first:

```bash
bash scripts/sh_scripts/build.sh
```

Active optimisation names:

```text
baseline
lower_triangle
upper_triangle
contiguous_access
cache_blocked_1
cache_blocked_2
```

## 1. Single Run

Run one optimisation for one matrix size.

```bash
./build/run/run_cholesky time baseline 512
./build/run/run_cholesky time baseline 512 results/raw/single_run_baseline_n512.csv
```

Arguments:

```text
./build/run/run_cholesky time <optimisation> <matrix_size> [raw_csv]
```

Expected output:

- stdout summary with optimisation, matrix size, elapsed time, and log-determinant checks
- optional raw CSV at the path you provide, for example:
  `results/raw/single_run_baseline_n512.csv`

## 2. Fixed-Size Comparison

Run all single-threaded optimisations for one matrix size.

```bash
./build/run/run_cholesky fixed-size 2000 3 results/raw/fixed_size_comparison_n2000_raw.csv
```

Arguments:

```text
./build/run/run_cholesky fixed-size <matrix_size> <repeats> <raw_csv>
```

Expected output:

- one raw CSV containing all methods, all repeats, and speedup versus baseline:
  `results/raw/fixed_size_comparison_n2000_raw.csv`

## 3. One Method Across Multiple Sizes

Run one optimisation across many matrix sizes and generate the scaling plots.

Shell workflow:

```bash
bash scripts/sh_scripts/run_matrix_size_graph.sh baseline 5 64 128 256 512 1024
```

Direct benchmark without plotting:

```bash
./build/run/run_cholesky scaling baseline 5 results/raw/baseline_matrix_size_graph.csv 64 128 256 512 1024
```

Arguments:

```text
./build/run/run_cholesky scaling <optimisation> <repeats> <raw_csv> <n1> [n2 ...]
```

Expected outputs:

- raw CSV:
  `results/raw/<optimisation>_matrix_size_graph.csv`
- summary CSV:
  `results/figures/<optimisation>_matrix_size_graph/summary_by_n.csv`
- runtime scaling plot:
  `results/figures/<optimisation>_matrix_size_graph/runtime_vs_n.png`
- cubic-scaling check plot:
  `results/figures/<optimisation>_matrix_size_graph/cubic_scaling_check.png`

Note:

- this graph is meant to show long-term scaling behaviour, so a wider spread of matrix sizes is usually better than several nearby values
- the runtime and cubic-scaling plots include error bars when repeats are greater than 1

## 4. All Methods Across Multiple Sizes

Run all single-threaded methods across many matrix sizes and generate the combined comparison plots.

Shell workflow:

```bash
bash scripts/sh_scripts/run_all_methods_graph.sh 3 2000 4000 6000
```

Expected outputs:

- one combined raw CSV:
  `results/raw/all_methods_graph.csv`
- combined runtime comparison plot:
  `results/figures/all_methods_graph/runtime_vs_n_by_method.png`
- combined speedup plot:
  `results/figures/all_methods_graph/speedup_vs_baseline.png`
- combined cubic-scaling check plot:
  `results/figures/all_methods_graph/cubic_scaling_check_by_method.png`

Note:

- all three comparison plots include error bars when repeats are greater than 1

## 5. Block-Size Sweep

Sweep the block size for one blocked optimisation and generate the block-size plots.

Shell workflow:

```bash
bash scripts/sh_scripts/run_block_size_sweep.sh cache_blocked_1 1000 5 16 24 32 48 64
```

Direct benchmark without plotting:

```bash
./build/run/run_cholesky block-size-sweep cache_blocked_1 1000 5 results/raw/cache_blocked_1_block_size_sweep_n1000.csv 16 24 32 48 64
```

Arguments:

```text
./build/run/run_cholesky block-size-sweep <optimisation> <matrix_size> <repeats> <raw_csv> <block_size1> [block_size2 ...]
```

Expected outputs:

- raw CSV:
  `results/raw/<optimisation>_block_size_sweep_n<matrix_size>.csv`
- summary CSV:
  `results/figures/<optimisation>_block_size_sweep_n<matrix_size>/summary_by_block_size.csv`
- runtime-vs-block-size plot:
  `results/figures/<optimisation>_block_size_sweep_n<matrix_size>/runtime_vs_block_size.png`
- speedup-vs-block-size plot:
  `results/figures/<optimisation>_block_size_sweep_n<matrix_size>/speedup_vs_block_size.png`

# CSD3 / Slurm

The Slurm scripts live in `scripts/slurm_scripts/`.

Single-thread experiments:

- `scripts/slurm_scripts/single_thread_block_size_sweep.slurm`
- `scripts/slurm_scripts/single_thread_fixed_size_comparison.slurm`
- `scripts/slurm_scripts/single_thread_scaling_all_methods.slurm`
- `scripts/slurm_scripts/single_thread_large_n_csvs.slurm`

OpenMP experiments:

- `scripts/slurm_scripts/openmp_scaling_with_baseline.slurm`
- `scripts/slurm_scripts/openmp_thread_count_sweep.slurm`
- `scripts/slurm_scripts/openmp_large_n_csvs.slurm`

Note:

- each Slurm script is written to be submitted on CSD3 with your own account and partition, for example:
  `sbatch --account=<ACCOUNT> --partition=<PARTITION> <script.slurm>`
- the scripts include comments where you may need to load your CSD3 Python/Conda environment for plotting
