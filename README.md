# C2 Coursework - Cholesky Decomposition

This project focuses on Cholesky decomposition, a matrix factorisation method developed by André-Louis Cholesky and published posthumously in 1924 (Cholesky, 1924). This method takes a Symmetric Positive Definite (SPD) matrix and decomposes it into the product of a lower triangular matrix and its transpose, allowing efficient and numerically stable computation (Golub & Van Loan, 2013; Higham, 2002). It is widely used in applications such as solving linear systems and statistical modelling. 

For a symmetric positive definite (SPD) matrix `A`, the Cholesky decomposition expresses `A` as the product of a triangular matrix and its transpose. In lower-triangular form, this is written as

\f[
A = L L^T
\f]

where `L` is a lower triangular matrix with strictly positive diagonal entries. The same decomposition can also be written in upper-triangular form as

\f[
A = R^T R
\f]

where `R = L^T`. This factorisation exists whenever `A` is symmetric and positive definite, and is unique when the diagonal entries are required to be positive.

In entry-wise form, the lower-triangular factor is defined by

\f[
L_{jj} = \sqrt{A_{jj} - \sum_{k=1}^{j-1} L_{jk}^{2}}, \qquad 1 \leq j \leq n
\f]

\f[
L_{ij} = \frac{A_{ij} - \sum_{k=1}^{j-1} L_{ik} L_{jk}}{L_{jj}}, \qquad i > j
\f]

\f[
L_{ij} = 0, \qquad i < j
\f]

In practice, these reduce to two core update rules used directly in the algorithm:

\f[
L_{jj} = \sqrt{A_{jj} - \sum_{k=1}^{j-1} L_{jk}^{2}}
\f]

\f[
L_{ij} = \frac{A_{ij} - \sum_{k=1}^{j-1} L_{ik} L_{jk}}{L_{jj}}, \qquad i > j
\f]

These expressions have a simple interpretation. Each diagonal entry is computed by taking the square root of what remains after subtracting contributions from previously computed columns. Each off-diagonal entry is then obtained by correcting the original value using a dot product of already known values, followed by normalisation. Entries above the diagonal are zero by construction.

This is the dense Banachiewicz form that we use as the mathematical basis for all implementations in this repository. This form naturally breaks our computation into three main steps

1. Factor the diagonal block  
2. Solve the panel block below the diagonal block  
3. Update the trailing submatrix  

This structure is preserved across all implementations, including the optimised and parallel versions. While the work may be reorganised (for example into blocks or tasks) to improve performance, the underlying mathematics remains unchanged. All implementations compute the same factorisation \f$ A = L L^T \f$ (or equivalently \f$ A = R^T R \f$), so correctness is unaffected — only the execution strategy differs.

An additional advantage of this formulation is that the factorisation can be performed in place. The original matrix storage is reused to store the result, meaning memory usage remains constant. This becomes particularly important for large matrices.

Cholesky decomposition is widely studied in numerical linear algebra, with many variants such as left-looking, right-looking, and blocked algorithms designed for modern hardware (Golub & Van Loan, 2013; Higham, 2002; Dongarra et al., 1990). Readers interested in further details are encouraged to consult these references.

This project implements several variations of Cholesky decomposition with the aim of improving runtime while maintaining correctness, these implementations are

- [`baseline`](src/00_cholesky_baseline.cpp): simple reference version, updates both triangles, easiest to reason about  
- [`lower_triangle`](src/01_cholesky_lower_triangle.cpp): computes the lower-triangular factor then mirrors it  
- [`upper_triangle`](src/02_cholesky_upper_triangle.cpp): computes the upper-triangular factor then mirrors it  
- [`contiguous_access`](src/03_cholesky_contiguous_access.cpp): single-threaded version arranged to improve memory access behaviour  
- [`cholesky_blocked_tile_kernels`](src/04_cholesky_blocked_tile_kernels.cpp): single-threaded blocked version with explicit tile structure  
- [`cholesky_blocked_tile_kernels_unrolled`](src/05_cholesky_blocked_tile_kernels_unrolled.cpp): blocked version with manual inner-loop unrolling  
- [`openmp_row_parallel_unblocked`](src/06_cholesky_openmp_row_parallel_unblocked.cpp): OpenMP version with row-level parallel work and no blocking  
- [`openmp_tile_parallel_blocked`](src/07_cholesky_openmp_tile_parallel_blocked.cpp): OpenMP blocked version with tile-parallel updates  
- [`openmp_block_row_parallel`](src/08_cholesky_openmp_block_row_parallel.cpp): OpenMP blocked version parallelised over block rows  
- [`openmp_tile_list_parallel`](src/09_cholesky_openmp_tile_list_parallel.cpp): OpenMP blocked version using an explicit tile work list  
- [`openmp_task_dag_blocked`](src/10_cholesky_openmp_task_dag_blocked.cpp): OpenMP blocked version using task dependencies  

---
## Features

The key features of this project are:

- SPD matrix generation based on the Gershgorin Circle Theorem, ensuring all generated inputs are valid for Cholesky decomposition  
- C++ implementations of multiple decomposition variants with no external dependencies (except OpenMP)  
- A benchmarking pipeline with four modes:  
  1. Method comparison  
  2. Matrix size variation  
  3. OpenMP thread count sweep  
  4. Block size sweep  

  Each mode optionally includes correctness checking  

- All benchmark modes output results as CSV files  
- Plotting support for:
  - runtime vs matrix size  
  - runtime vs block size  
  - runtime vs thread count  
  - multi-method matrix size comparison  

- A comprehensive test suite covering:
  - mathematical correctness  
  - cross-method consistency  
  - OpenMP robustness  
  - command-line parsing  
  - end-to-end execution  

- Documentation generated using Doxygen  

---
## Project Structure

The repository is organised into a set of folders separating implementations, benchmarking, testing, and utilities. A simplified overview is shown below:

```bash
├── build
│   ├── cmake_install.cmake
│   ├── CMakeCache.txt
│   ├── CMakeDoxyfile.in
│   ├── CMakeDoxygenDefaults.cmake
│   ├── CMakeFiles
│   ├── compile_commands.json
│   ├── CTestTestfile.cmake
│   ├── DartConfiguration.tcl
│   ├── Doxyfile.docs
│   ├── Makefile
│   ├── performance_tests
│   ├── run
│   ├── src
│   ├── test
│   └── Testing
├── CMakeFiles
│   └── CMakeSystem.cmake
├── CMakeLists.txt
├── CMakePresets.json
├── environment.yml
├── include
│   └── cholesky_decomposition.h
├── Makefile
├── MPhil_DIS_C2_coursework.pdf
├── performance_tests
│   ├── 01_perf_method_compare.cpp
│   ├── 02_perf_thread_count_sweep.cpp
│   ├── 03_perf_block_size_sweep.cpp
│   ├── 04_perf_matrix_size_sweep.cpp
│   ├── CMakeLists.txt
│   ├── perf_helpers.cpp
│   ├── perf_helpers.h
│   ├── perf_modes.h
│   ├── runtime_cholesky.cpp
│   └── runtime_cholesky.h
├── plot
│   ├── __pycache__
│   ├── cholesky_plotter.py
│   └── pythonStyle.mplstyle
├── README copy.md
├── README.md
├── report
├── requirements-plot.txt
├── results
│   ├── Archive
│   ├── figures
│   └── raw
├── run
│   ├── CMakeLists.txt
│   └── run_cholesky.cpp
├── run_guides
│   ├── block_size_sweep.md
│   ├── matrix_size_sweep.md
│   ├── method_compare.md
│   └── thread_count_sweep.md
├── scripts
│   ├── openmp
│   └── single_thread
├── src
│   ├── 00_cholesky_baseline.cpp
│   ├── 01_cholesky_lower_triangle.cpp
│   ├── 02_cholesky_upper_triangle.cpp
│   ├── 03_cholesky_contiguous_access.cpp
│   ├── 04_cholesky_blocked_tile_kernels.cpp
│   ├── 05_cholesky_blocked_tile_kernels_unrolled.cpp
│   ├── 06_cholesky_openmp_row_parallel_unblocked.cpp
│   ├── 07_cholesky_openmp_tile_parallel_blocked.cpp
│   ├── 08_cholesky_openmp_block_row_parallel.cpp
│   ├── 09_cholesky_openmp_tile_list_parallel.cpp
│   ├── 10_cholesky_openmp_task_dag_blocked.cpp
│   ├── cholesky_decomposition.cpp
│   ├── cholesky_helpers.h
│   ├── cholesky_versions.h
│   └── CMakeLists.txt
├── test
│   ├── __pycache__
│   ├── CMakeLists.txt
│   ├── gtest
│   ├── test_cli_gtest.cpp
│   ├── test_correctness_gtest.cpp
│   ├── test_csv_contract_gtest.cpp
│   ├── test_integration_gtest.cpp
│   ├── test_openmp_gtest.cpp
│   ├── test_plot_gtest.cpp
│   ├── test_regression_gtest.cpp
│   ├── test_suite_helpers.cpp
│   ├── test_suite_helpers.h
│   └── test_unit_gtest.cpp
└── utils
    ├── CMakeLists.txt
    ├── matrix.cpp
    ├── matrix.h
    ├── timer.cpp
    └── timer.h
```

The key directories are:

- `include/`  
  Public headers defining the interface, available implementations, and runtime configuration options  

- `src/`  
  All Cholesky implementations, including both single threaded and multi-threaded (using OpenMP) implementations

- `performance_tests/`  
  Benchmark modes (method comparison, scaling, block size, thread count) and CSV/statistics helpers  

- `run/`  
  Command-line tool (`run_cholesky`) used to execute benchmarks  

- `test/`  
  Full GoogleTest suite covering correctness, regression, OpenMP behaviour, and integration  

- `utils/`  
  Matrix generation, validation, and timing utilities used across the project  

- `plot/`  
  Python plotting scripts for visualising benchmark results  

- `scripts/`  
  Helper scripts for running experiments, including Slurm job scripts for CSD3  

- `run_guides/`  
  Documentation on how to use each benchmark mode  

- `results/`  
  Output data (CSV) and generated figures  

- `report/`  
  Coursework report


---
## Requirements

To build and run the project, the following are required:

- C++17-compatible compiler (e.g. GCC or Clang)  
- CMake  
- Make  
- OpenMP support (for parallel implementations)
- LAPACK (used for correctness reference calculations)  

If building the test suite GoogleTest is required and plotting requires Python and packaged listed in `requirements-plot.txt` or `environment.yml`. A simple Python setup using a virtual environment is

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-plot.txt
```

Or using Conda:
```bash
conda env create -f environment.yml
conda activate c2-coursework
```

On macOS, the project assumes Homebrew-installed `LLVM` and `libomp` for OpenMP support. The provided Makefile and CMake presets are configured for this setup. On CSD3 (HPC), the Slurm scripts load the required modules automatically.

---
## Build

The project provides several builds dependent on the level of testing required:

- `make build`  
  Builds the project only  

- `make test-fast`  
  Builds the project and runs a small, fast subset of tests  

- `make test`  
  Builds the project and runs the full test suite  

- `make verify`  
  Alias for the full build and test workflow  

Note these commands should be done in the root of the project and the default build uses compiler optimisation and native architecture tuning.

---
## Running

There are four main benchmark modes which are recommended. These are:

  - `method-compare`: same `n`, same repeats, several methods side by side
  - `matrix-size-sweep`: one method across several matrix sizes
  - `block-size-sweep`: one blocked method across several block sizes
  - `thread-count-sweep`: OpenMP methods across several thread counts

Once built, these can be run with the following general syntax

```bash
./build/run/run_cholesky method-compare <n> <repeats> <raw_csv> [methods...] [--threads N] [--block-size N] [--block-size-for METHOD=SIZE ...] [--correctness]
./build/run/run_cholesky matrix-size-sweep <optimisation> <repeats> <raw_csv> <n1> [n2 ...] [--threads N] [--block-size N] [--correctness]
./build/run/run_cholesky block-size-sweep <optimisation> <n> <repeats> <raw_csv> <block_size1> [block_size2 ...] [--threads N] [--correctness]
./build/run/run_cholesky thread-count-sweep <n> <repeats> <raw_csv> --threads <t1> [t2 ...] [--methods <m1> [m2 ...]] [--block-size N] [--correctness]
```

In addition there is a extra utility mode:
  
  - `matrix-generator-compare`: compare the matrix generation paths and record correctness-style diagnostics

---
## Example

Here we show two example commands, one single threaded and one multi-threaded. For further information about these
running modes and more example commands, please see the `run_guides`:

  - [Method Compare](run_guides/method_compare.md)
  - [Matrix-Size Sweep](run_guides/matrix_size_sweep.md)
  - [Block-Size Sweep](run_guides/block_size_sweep.md)
  - [Thread-Count Sweep](run_guides/thread_count_sweep.md)

1. small single-thread example, fixed-size comparison:

```bash
./build/run/run_cholesky \
  method-compare \
  2000 \
  3 \
  results/raw/method_compare_n2000.csv \
  baseline \
  contiguous_access \
  cholesky_blocked_tile_kernels \
  --block-size 32 \
  --correctness
```

2. Multi-thread OpenMP example:

```bash
./build/run/run_cholesky \
  method-compare \
  10000 \
  3 \
  results/raw/openmp_method_compare_n10000.csv \
  openmp_row_parallel_unblocked \
  openmp_tile_parallel_blocked \
  openmp_tile_list_parallel \
  --threads 8 \
  --block-size 32 \
  --block-size-for openmp_tile_list_parallel=64 \
  --correctness
```

---
## Testing

The project uses GoogleTest for all test cases, with CTest handling test discovery and execution.

GoogleTest is used because it provides a simple and reliable way to write structured C++ tests, with clear assertions, good failure reporting, and easy grouping of related tests. It also integrates cleanly with CMake and CTest, making it straightforward to run both small unit tests and larger integration tests in a consistent way.

The current test suite contains 33 tests, organised into the following groups:

- `unit` (10 tests)  
- `correctness` (3 tests)  
- `openmp` (3 tests)  
- `regression` (3 tests)  
- `integration` (10 tests)  
- `plot` (4 tests)  

The test suite covers:

- Small worked examples and basic matrix properties  
- Reconstruction and log-determinant correctness across all implementations  
- Behaviour under awkward block sizes and varying OpenMP thread counts  
- Method name parsing, including aliases and invalid inputs  
- CSV output format, including headers and correctness fields  
- End-to-end execution of all benchmark modes  
- Incremental CSV writing during long runs  
- Basic plotting pipeline checks (smoke tests)  

Below are useful commands for running the test suite

1. List all available tests:

```bash
ctest --test-dir build -N
```

2. Run the full test suite:

```bash
ctest --preset macos-test-full --output-on-failure
```

3. Run a faster subset of tests:

```bash
ctest --preset macos-test-fast --output-on-failure
```

Please note the OpenMP tests check correctness rather than performance, so passing them does not guarantee good scaling. In addition plot tests are lightweight smoke tests and only verify that the plotting pipeline runs without errors, not that the figures are visually correct.  

---
## Using This Project in A Performative Manner

Looking beyond single-threaded optimisations, parallel performance is always a trade-off between doing useful work and paying overheads. Simply adding more threads, or increasing the amount of work per thread, does not guarantee better performance. This project has highlighted that quite clearly. Two of the most important bottlenecks we observed are


- **Memory access patterns dominating performance**
Although several improvements were made to encourage contiguous memory access and better cache blocking, performance is still often limited by how data is accessed rather than how much computation is performed. Even when there is plenty of arithmetic work available, poorly structured memory access (for example, low locality or weak cache reuse) can cause the processor to spend a significant amount of time waiting for data. In practice, this means that improving memory layout and access patterns can be just as important as improving the underlying algorithm itself.


- **Memory bandwidth limits performance at large matrix sizes**
As n grows (particularly beyond ~50,000), the computation becomes increasingly memory-bound. Although there is more work available, the rate at which data can be moved from memory becomes the limiting factor. What you might expect is that larger problems scale better with more threads. In practice, performance gains start to diminish, and in some cases can even get worse. This highlights that further improvements need to focus on memory locality (such as better blocking and cache reuse), rather than simply increasing parallelism.

Other observed bottlenecks include:
-	Cache efficiency being highly sensitive to block size
- Synchronisation overhead between OpenMP regions
- Poor load balance leading to idle threads
- NUMA effects becoming visible at large matrix sizes

Motivated by these bottlenecks, without modifying the code, one can perform lots of useful performance analysis such as

- **Running on different HPC systems or partitions**  
  Although in this project we have run single-threaded optimisations across different partitions on CSD3, the high number of threads necessary (and time-limit) meant it was unfeasible to run multi-threaded experiments particularly for high matrix sizes. Different machines have different cache sizes, memory bandwidth, and core counts. This means that “optimal” settings (block size, thread count) are not universal.Although we would similar trends across machines, optimal parameters can shift significantly depend on machine. Things like the physical distance from the CPU to the cache differ. This will demonstrates that performance tuning is hardware-dependent, and helps to validate whether the implementation are robhust across a wider range of systems


- **Exploring block size vs thread count interaction**  
  Block size and thread count are often treated as separate tuning parameters, but in practice they are closely linked. A block size that works well for a small number of threads (for example 8) can perform noticeably worse when scaled up to 76 threads, because the amount of work per thread and cache behaviour both change. We might expect there to be a single best block size, but the results show that this is not the case. The optimal choice depends on both the matrix size and the number of threads being used. This can be seen clearly in the multithreaded results at \(n = 5000\), where increasing the thread count from around \(50\) up to \(76\) only gives a very small improvement in performance. This suggests that the system is already close to its effective limit, and that simply adding more threads is not enough; instead, block size and workload need to be tuned together to maintain good efficiency at higher thread counts.

Other possible experiments include:

- Testing different OpenMP settings (`OMP_PROC_BIND`, `OMP_PLACES`, scheduling)  
- Performing more detailed parameter sweeps  
- Analysing efficiency (speedup vs threads) instead of just runtime  
- Comparing behaviour across matrix size regimes  
- Observing where scaling saturates  

To improve performance further, more substantial changes to the implementation are needed. Two of the most promising directions are:

- **Improving the task-based (DAG) implementation**  
Improving the task-based (DAG) implementation is largely about getting the balance right between useful work and overhead. At the moment, tasks are often too small or not well balanced, which means the cost of creating and scheduling tasks can outweigh the actual computation being done. In theory, task-based parallelism should scale very well because it allows flexible scheduling and better load balance than simple loops. In practice, if the tasks are too fine-grained, the overhead dominates and performance suffers. Increasing the amount of work per task and reducing scheduling overhead would likely improve both thread utilisation and overall efficiency.


- **BLAS-inspired or more compute-dense updates**  
A second clear direction is moving towards more BLAS-inspired, compute-dense updates. In particular, restructuring the trailing update to look more like matrix–matrix operations can significantly improve cache reuse and allow better vectorisation. We might expect this to behave similarly to the current implementation, just written differently, but in reality this kind of change can lead to large performance gains because it makes much better use of the memory hierarchy. This is exactly the approach used in highly optimised libraries such as LAPACK, and it is one of the main reasons they outperform simpler loop-based implementations.

Other possible extensions include:

- Hybrid parallelism (MPI + OpenMP) for multi-node scaling  
- NUMA-aware memory placement strategies  
- Alternative blocking strategies (adaptive or hierarchical)  
- Improved scheduling strategies (dynamic or guided)  

Finally some general advice for someone wanting to extend this program. The best way to begin is to start small and build up. Use small matrix sizes first to check everything is working properly, and run `test-fast` early to catch obvious issues before committing to longer runs. It helps to begin with the simpler implementations (baseline, then blocked, then OpenMP) so you get a feel for how the algorithm behaves before adding parallel complexity (and do not waste CPU hours).

Keep correctness checks turned on while developing, and only switch them off once you are confident everything is stable and you are focusing purely on performance. Increase matrix size gradually and watch how performance changes, rather than jumping straight to very large cases.

Block size and thread count should always be tuned experimentally, not just chosen from theory. Use the provided `block-size-sweep` and `thread-count-sweep` modes to explore this properly — the optimal settings often depend on both the matrix size and the hardware.

When running on the cluster, make sure your code is actually using multiple threads. SSH into the compute node and check with `htop`; if you only see partial CPU usage or uneven load, something is likely wrong with the parallelisation. Finally, run multiple repeats for stable timings, avoid running other workloads at the same time, and use the plotting tools to look at trends rather than relying on single measurements.

---
## Tagged Commits

As per instructions, Below are the tagged commits for the project

| Tag    | Commit   | Description |
| ---    | ---      |  --- |
| `v0.1` | `TBD`    |  XXX |
| `v0.2` | `TBD`    |  XXX |
| `v0.3` | `TBD`    |  XXX |
| `v0.4` | `TBD`    |  XXX |
| `v1.0` | `TBD`    |  XXX |

---
## Documentation

The repo has Doxygen wired into the normal CMake build, and this `README.md` is used as the main page for the generated site. After configuring the build, generate the docs with:

```bash
cmake --build build --target docs
```

The HTML output is written to `build/docs/html/index.html`. Doxygen covers the reusable C++ library API (`public library API`), the implementation files, the benchmark layer, and the command-line entry points (`CLI entry points`). The test tree is intentionally excluded so the generated docs stay focused on the actual project code rather than the verification harness.

---
## CSD3

Cluster runs in this repository target the `icelake` partition unless stated otherwise. The `icelake` partition on CSD3 consists of Intel Xeon Ice Lake nodes with 76 physical CPU cores per node and 256 GB RAM per node 

Thus the typical OpenMP Slurm configuration used in this project is:

- 1 node  
- 1 task  
- 76 CPUs per task  (note this varies dependent on test)

Slurm job scripts are located in:

- `scripts/slurm_scripts/`  

Unless explicitly stated, all report-style benchmark results should be assumed to come from CSD3 `icelake` runs.

For local development, the project was tested on an Apple Silicon laptop with M3 Pro chip and 18 GB memory  

Local runs are primarily used for development, debugging, and smaller-scale experiments unless otherwise stated.

---
## AI Declaration





---
## References

- Cholesky, A.-L. *Sur la résolution numérique des systèmes d’équations linéaires*. Bulletin Géodésique, 1924  
- Golub, G. H., & Van Loan, C. F. *Matrix Computations* (4th ed.). Johns Hopkins University Press, 2013  
- Higham, N. J. *Accuracy and Stability of Numerical Algorithms* (2nd ed.). SIAM, 2002 
- Dongarra, J., Duff, I., Sorensen, D., & Van der Vorst, H. *Numerical Linear Algebra for High-Performance Computers*. SIAM, 1998  
