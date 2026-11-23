# Assignment 04: Distributed Matrix Multiplication

Empirical investigation of dense matrix multiplication scaling ($C = A \times B$, `float64`) using **Dask (`dask.array` & `dask.distributed`)**, evaluating computation time, inter-worker communication, serialization overhead, overlapping execution phases, and memory footprint.

## Partitioning Strategies

* **`1d_row_broadcast`:** Matrix $A$ is split horizontally into $P$ row blocks (`(N/P, N)`), while $B$ is broadcast intact (`(N, N)`) to all workers. Eliminates reduction/shuffle communication at the cost of replicating $B$ across worker memory.
* **`2d_block`:** Both $A$ and $B$ are tiled into $P \times P$ sub-blocks (`(N/P, N/P)`), reducing per-task memory footprint but introducing $O(P^2)$ block transfers and tree-reduction (`shuffle`) overhead.

## Execution Environment & Scope Limitation

* **Hardware:** Intel Core i7-10750H (6 physical cores / 12 logical threads), 16 GB DDR4 RAM, Windows 11.
* **Software:** Python 3.13, NumPy (single-threaded BLAS per worker via `OPENBLAS_NUM_THREADS=1`), Dask Distributed.
* **Cluster Topology:** Local multi-process cluster (`LocalCluster(processes=True, threads_per_worker=1)`) across $P \in \{1, 2, 4, 6\}$ independent OS worker processes on a single physical machine.
* **Environment Limitation:** Multi-process local execution accurately isolates framework scheduling, serialization, and IPC/loopback TCP overhead, but does not measure physical inter-machine network latency or aggregate multi-node RAM capacity.

## Key Empirical Findings

* **Strong Scaling ($N = 4000$, $128\text{ GFLOP}$):** `1d_row_broadcast` scales from **$3.16\text{ s}$ ($P=1$)** down to **$2.30\text{ s}$ ($P=6$)**, whereas `2d_block` reaches its optimum at **$2.39\text{ s}$ ($P=4$)** before degrading to **$2.92\text{ s}$ ($P=6$)** due to shuffle congestion.
* **Overlapping Phases (Non-Additivity):** At $N = 4000, P = 6$ (`2d_block`), cumulative task transfer activity reaches **$9.51\text{ s}$** while actual end-to-end wall-clock execution is **$2.92\text{ s}$**, confirming concurrent pipelining across workers.
* **Paradigm Trade-off ($N = 2000$):** Shared-memory parallelism (`0.118 s`) outperforms distributed Dask (`0.464 s` in 1D) because BLAS computation is fast enough that IPC serialization and scheduler orchestration dominate total runtime.

## Repository Structure

```text
Assignment04_Morales_Repository/
├── src/
│   ├── __init__.py
│   ├── baselines.py               # Validated triple-loop, NumPy BLAS & shared-memory references
│   └── distributed_matmul.py      # 1D-Row & 2D-Block Dask strategies + task-stream telemetry
├── scripts/
│   ├── generate_matrices.py       # Deterministic float64 matrix generator (.npy)
│   ├── validate_correctness.py    # Numerical verification (rtol < 1e-10)
│   ├── run_benchmarks.py          # Paradigm comparison, strong scaling & weak scaling suite
│   └── plot_results.py            # High-resolution Matplotlib figures (300 DPI)
├── data/                          # Binary .npy datasets (git-ignored)
├── results/
│   ├── paradigm_comparison.csv
│   ├── strong_scaling.csv
│   ├── weak_scaling.csv
│   ├── plots/                     # Exported PNG figures
│   └── reports/                   # Dask HTML performance reports
├── .gitignore
├── requirements.txt
└── README.md
```

## Reproducibility
```
python -m venv .venv
.venv\Scripts\activate.bat
pip install -r requirements.txt

python scripts\generate_matrices.py
python scripts\validate_correctness.py

set OPENBLAS_NUM_THREADS=1
set MKL_NUM_THREADS=1
set OMP_NUM_THREADS=1
python scripts\run_benchmarks.py --runs 5 --workers 1 2 4 6
python scripts\plot_results.py
```