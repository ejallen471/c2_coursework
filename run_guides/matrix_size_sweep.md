# Matrix-Size Sweep

Use `matrix-size-sweep` to study how one method scales as the matrix size changes.

Build first (without testing):

```bash
make build
```

For the full build-plus-test workflow:

```bash
make verify
```

Once built, the general syntax is:

```bash
./build/run/run_cholesky matrix-size-sweep <method> <repeats> <raw_csv> <n1> [<n2> <n3> ...] [--threads N] [--block-size N] [--correctness]
```

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

Outputs:

- Raw CSV containing `method`, `n`, `repeat`, and `elapsed_seconds`
- Additional correctness fields are included in the raw CSV when `--correctness` is enabled: `logdet_library`, `logdet_factor`, and `relative_difference_percent`
- Summary CSV reporting `elapsed_median`, `elapsed_mean`, and `elapsed_error`

## Single-thread example

```bash
./build/run/run_cholesky \
  matrix-size-sweep \
  cholesky_blocked_tile_kernels_unrolled \
  5 \
  results/raw/cholesky_blocked_tile_kernels_unrolled_scaling.csv \
  256 512 1000 2000 4000 8000 \
  --block-size 32 \
  --correctness
```

## OpenMP example

```bash
./build/run/run_cholesky \
  matrix-size-sweep \
  openmp_tile_list_parallel \
  5 \
  results/raw/openmp_tile_list_parallel_scaling.csv \
  1000 2000 5000 10000 \
  --threads 10 \
  --block-size 32 \
  --correctness
```

`--threads` should not exceed the number of available hardware threads on the system.

## Plot one single method

Plotting is performed with Python. Ensure the appropriate environment is activated with the required dependencies installed. Please see the `README.md` for further information

```bash
python plot/cholesky_plotter.py \
  matrix-size \
  results/raw/openmp_tile_list_parallel_scaling.csv \
  results/figures/openmp_tile_list_parallel_scaling
```

## Plot several methods together

Run `matrix-size-sweep` once per method, then pass the CSVs to the comparison plotter:

```bash
python plot/cholesky_plotter.py \
  matrix-size-comparison \
  results/figures/all_methods_scaling \
  results/raw/baseline_scaling.csv \
  results/raw/contiguous_access_scaling.csv \
  results/raw/cholesky_blocked_tile_kernels_unrolled_scaling.csv
```

## CSD3

This experiment can also be run on CSD3 using Slurm scripts. The command is:

```bash
sbatch scripts/slurm_scripts/single_thread_matrix_size_sweep.slurm
```

```bash
sbatch scripts/slurm_scripts/openmp_matrix_size_sweep.slurm
```

Before running, check that the Slurm directives (such as partition, time, and CPU allocation) are appropriate for the target CSD3 environment.
