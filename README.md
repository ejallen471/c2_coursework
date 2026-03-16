# C2 Coursework

This project benchmarks several Cholesky factorisation implementations for dense symmetric positive definite matrices. It includes single-threaded variants, cache-blocked variants, and OpenMP variants, together with local shell workflows, CSD3 Slurm workflows, CSV output, and plotting scripts.

---
## Features

- Multiple Cholesky implementations in C++
- Unified benchmark runner with `time`, `fixed-size`, `scaling`, and `block-size-sweep` modes
- Local shell workflows for single-thread and OpenMP experiments
- CSV outputs for benchmarking and correctness checks
- Python plotting scripts for scaling, speedup, and block-size analysis
- Tests for correctness, parsing, matrix generation, and runner output

---
## Installation

Build the standard single-thread executable:

```bash
bash scripts/sh_scripts/build.sh
```

Build the OpenMP executable:

```bash
bash scripts/sh_scripts/build_openmp.sh
```

For plotting locally, create a Python environment and install:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-plot.txt
```

---
## Testing

Build scripts compile the test executables, but they do not run the tests automatically.

Run the standard test suite with:

```bash
bash scripts/sh_scripts/build.sh
ctest --test-dir build --output-on-failure
```

Run the OpenMP-enabled test suite with:

```bash
bash scripts/sh_scripts/build_openmp.sh
ctest --test-dir build_openmp --output-on-failure
```

--- 
## Usage

The main local workflows are:

- Single-thread fixed-size comparison  
  Run all single-thread implementations at one matrix size and write one comparison CSV.
  Command:
  ```bash
  ./build/run/run_cholesky fixed-size <matrix_size> <repeats> <raw_csv>
  ```
  Example:
  ```bash
  ./build/run/run_cholesky fixed-size 2000 3 results/raw/fixed_size_comparison_n2000_raw.csv
  ```

- Single-thread one-method scaling  
  Run one single-thread implementation across several matrix sizes and generate plots.
  Command:
  ```bash
  bash scripts/sh_scripts/run_matrix_size_graph.sh <optimisation> <repeats> <n1> [n2 ...]
  ```
  Example:
  ```bash
  bash scripts/sh_scripts/run_matrix_size_graph.sh baseline 5 64 128 256 512 1024
  ```

- Single-thread block-size sweep  
  Sweep the block size for one blocked single-thread implementation and generate plots.
  Command:
  ```bash
  bash scripts/sh_scripts/run_block_size_sweep.sh <optimisation> <matrix_size> <repeats> <block_size1> [block_size2 ...]
  ```
  Example:
  ```bash
  bash scripts/sh_scripts/run_block_size_sweep.sh cache_blocked_1 1000 5 16 24 32 48 64
  ```

- OpenMP fixed-size comparison  
  Run the selected OpenMP fixed-size methods at one matrix size, then write one comparison CSV.
  Command:
  ```bash
  OPENMP_THREADS=<threads> bash scripts/sh_scripts/run_openmp_fixed_size_comparison.sh <matrix_size> <repeats> [raw_csv] [--methods <method1> [method2 ...]]
  ```
  Example:
  ```bash
  OPENMP_THREADS=8 bash scripts/sh_scripts/run_openmp_fixed_size_comparison.sh 2000 1
  ```

- OpenMP matrix-size scaling  
  Run `baseline` and all OpenMP implementations across several matrix sizes and generate comparison plots.
  Command:
  ```bash
  OPENMP_THREADS=<threads> bash scripts/sh_scripts/run_openmp_methods_graph.sh <repeats> <n1> [n2 ...]
  ```
  Example:
  ```bash
  OPENMP_THREADS=8 bash scripts/sh_scripts/run_openmp_methods_graph.sh 3 512 1024 2000 4096
  ```

- OpenMP thread-count sweep  
  Run the OpenMP implementations at one matrix size across several thread counts and generate a thread-scaling plot.
  Command:
  ```bash
  bash scripts/sh_scripts/run_openmp_thread_count_sweep.sh <matrix_size> <repeats> <threads1> [threads2 ...]
  ```
  Example:
  ```bash
  bash scripts/sh_scripts/run_openmp_thread_count_sweep.sh 2000 3 1 2 4 8
  ```

For a direct single run without a wrapper, you can still call the benchmark runner itself:

Command:
```bash
./build/run/run_cholesky time <optimisation> <matrix_size> [raw_csv]
```

Example:
```bash
./build/run/run_cholesky time baseline 512
```

---
## Report

A write up of our investigation and results can be found in report/c2_report.pdf


---
## Documentation


---
## Contributing


---
## License

---
## AI Declaration
