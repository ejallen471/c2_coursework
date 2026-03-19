# Block-Size Sweep

Use `block-size-sweep` to tune one blocked method at a fixed matrix size across several candidate block sizes.

Build first (without testing):

```bash
make build
```

For the full build-plus-test workflow:

```bash
make verify
```

Once successfully built, the general syntax is:

```bash
./build/run/run_cholesky block-size-sweep <method> <n> <repeats> <raw_csv> <block_size1> [block_size2 ...] [--threads N] [--correctness]
```

Only the first positional argument after `block-size-sweep` is a method name.
After `raw_csv`, every remaining positional argument is a block size.

OpenMP blocked methods in this mode require `--threads N`.


Supported blocked methods:

- `cholesky_blocked_tile_kernels`
- `cholesky_blocked_tile_kernels_unrolled`
- `openmp_tile_parallel_blocked`
- `openmp_block_row_parallel`
- `openmp_tile_list_parallel`
- `openmp_task_dag_blocked`

Outputs:

- Raw CSV containing `method`, `n`, `block_size`, `repeat`, and `elapsed_seconds`
- Additional correctness fields are included in the raw CSV when `--correctness` is enabled: `logdet_library`, `logdet_factor`, and `relative_difference_percent`
- Summary CSV reporting `elapsed_median`, `elapsed_mean`, and `elapsed_error`

## Single-thread example

```bash
./build/run/run_cholesky \
  block-size-sweep \
  cholesky_blocked_tile_kernels \
  2000 \
  5 \
  results/raw/cholesky_blocked_tile_kernels_block_size_sweep_n2000.csv \
  16 24 32 64 96 \
  --correctness
```

## OpenMP example

```bash
OMP_PROC_BIND=close \
OMP_PLACES=cores \
./build/run/run_cholesky \
  block-size-sweep \
  openmp_tile_list_parallel \
  10000 \
  5 \
  results/raw/openmp_tile_list_parallel_block_size_sweep_n10000.csv \
  16 32 64 96 128 \
  --threads 10 \
  --correctness
```

`--threads` should not exceed the number of available hardware threads on the system.

## Plotting

Plotting is performed with Python. Ensure the appropriate environment is activated with the required dependencies installed. Please see the `README.md` for further information

```bash
python plot/cholesky_plotter.py \
  block-size \
  results/raw/openmp_tile_list_parallel_block_size_sweep_n10000.csv \
  results/figures/openmp_tile_list_parallel_block_size_sweep_n10000
```

## CSD3

This experiment can also be run on CSD3 using Slurm scripts. The command is:

```bash
sbatch scripts/slurm_scripts/single_thread_block_size_sweep.slurm
```

```bash
sbatch scripts/slurm_scripts/openmp_block_size_sweep.slurm
```

Before running, check that the Slurm directives (such as partition, time, and CPU allocation) are appropriate for the target CSD3 environment.

The scripts can be configured to run multiple methods and generate a single combined plot. Refer to the corresponding Slurm scripts for configuration details.
