# Thread-Count Sweep

Use `thread-count-sweep` to compare OpenMP runtime across several thread counts.

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
./build/run/run_cholesky thread-count-sweep <n> <repeats> <raw_csv> --threads <t1> [t2 ...] [--methods <m1> [m2 ...]] [--block-size N] [--correctness]
```

If `--methods` is omitted, the runner uses all OpenMP methods supported by this mode.

`--threads` controls the sweep directly and should be treated as the single source of truth for thread count in this mode.

Blocked OpenMP methods in this mode require a shared `--block-size N`.

Supported methods:

- `openmp_row_parallel_unblocked`
- `openmp_tile_parallel_blocked`
- `openmp_block_row_parallel`
- `openmp_tile_list_parallel`
- `openmp_task_dag_blocked`

Outputs:

- Raw CSV containing `method`, `n`, `threads`, `repeat`, and `elapsed_seconds`
- Additional correctness fields are included in the raw CSV when `--correctness` is enabled: `logdet_library`, `logdet_factor`, and `relative_difference_percent`
- Summary CSV reporting `elapsed_median`, `elapsed_mean`, and `elapsed_error`

## Example

```bash
./build/run/run_cholesky \
  thread-count-sweep \
  10000 \
  5 \
  results/raw/openmp_thread_count_sweep_n10000.csv \
  --threads 1 2 4 8 16 32 \
  --block-size 32 \
  --methods openmp_row_parallel_unblocked openmp_tile_parallel_blocked openmp_block_row_parallel openmp_tile_list_parallel openmp_task_dag_blocked \
  --correctness
```

## Plot the sweep

Plotting is performed with Python. Ensure the appropriate environment is activated with the required dependencies installed. Please see the `README.md` for further information

```bash
python plot/cholesky_plotter.py \
  thread-count \
  results/raw/openmp_thread_count_sweep_n10000.csv \
  results/figures/openmp_thread_count_sweep_n10000
```

## CSD3

This experiment can also be run on CSD3 using Slurm scripts. The command is:

```bash
sbatch scripts/slurm_scripts/openmp_thread_count_sweep.slurm
```

Before running, check that the Slurm directives (such as partition, time, and CPU allocation) are appropriate for the target CSD3 environment.
