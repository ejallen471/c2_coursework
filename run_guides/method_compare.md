# Method Compare

Use `method-compare` to compare multiple Cholesky implementations on the same matrix size.

Build first:

```bash
make build
```

For the full build-plus-test workflow:

```bash
make verify
```

After building, the general syntax to run is:

```bash
./build/run/run_cholesky method-compare <n> <repeats> <raw_csv> [methods...] [--threads N] [--block-size N] [--block-size-for METHOD=SIZE ...] [--correctness]
```

If the method list is omitted, the runner uses all methods supported by this mode.

Supported methods:

- `baseline`
- `lower_triangle`
- `upper_triangle`
- `contiguous_access`
- `cholesky_blocked_tile_kernels`
- `cholesky_blocked_tile_kernels_unrolled`
- `openmp_row_parallel_unblocked`
- `openmp_tile_parallel_blocked`
- `openmp_block_row_parallel`
- `openmp_tile_list_parallel`
- `openmp_task_dag_blocked`

Tuning controls:

- OpenMP methods require `--threads N`
- Blocked methods require `--block-size N`
- Per-method block-size overrides use repeated `--block-size-for METHOD=SIZE`

Outputs:

- Raw CSV containing `method`, `n`, `repeat`, `elapsed_seconds`, and `speedup_factor_vs_baseline`
- Additional correctness fields are included in the raw CSV when `--correctness` is enabled: `logdet_library`, `logdet_factor`, and `relative_difference_percent`
- This mode does not write a summary CSV

## Single-thread example

```bash
./build/run/run_cholesky \
  method-compare \
  2000 \
  5 \
  results/raw/fixed_size_comparison_n5000.csv \
  baseline lower_triangle upper_triangle contiguous_access cholesky_blocked_tile_kernels cholesky_blocked_tile_kernels_unrolled \
  --block-size 32 \
  --correctness
```

## OpenMP example

```bash
./build/run/run_cholesky \
  method-compare \
  5000 \
  5 \
  results/raw/openmp_fixed_size_comparison_n5000.csv \
  baseline openmp_row_parallel_unblocked openmp_tile_parallel_blocked openmp_block_row_parallel openmp_tile_list_parallel openmp_task_dag_blocked \
  --threads 10 \
  --block-size 16 \
  --block-size-for openmp_tile_list_parallel=64 \
  --block-size-for openmp_task_dag_blocked=128 \
  --correctness
```

## CSD3

This experiment can also be run on CSD3 using Slurm scripts. The command is:

```bash
sbatch scripts/slurm_scripts/single_thread_fixed_runs.slurm
```

```bash
sbatch scripts/slurm_scripts/openmp_fixed_runs.slurm
```

Before running, check that the Slurm directives (such as partition, time, and CPU allocation) are appropriate for the target CSD3 environment.
