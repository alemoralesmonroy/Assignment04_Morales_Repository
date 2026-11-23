# Assignment 04: Distributed Matrix Multiplication

Empirical investigation of dense matrix multiplication scaling ($C = A \times B$, `float64`) using **Dask (`dask.array` & `dask.distributed`)**, evaluating computation time, inter-worker communication, serialization overhead, overlapping execution phases, and memory footprint.

## Execution Environment & Scope Limitation

* **Hardware:** Intel Core i5-10400F (6 physical cores / 12 logical threads), 16 GB DDR4 RAM, Windows 11.
* **Software:** Python 3.13, NumPy (single-threaded BLAS per worker via `OPENBLAS_NUM_THREADS=1`), Dask Distributed.
* **Cluster Topology:** Local multi-process cluster (`LocalCluster(processes=True, threads_per_worker=1)`) across $P \in \{1, 2, 4, 6\}$ independent OS worker processes on a single physical machine.
* **Environment Limitation:** Multi-process local execution accurately isolates framework scheduling, serialization, and IPC/loopback TCP overhead, but does not measure physical inter-machine network latency or aggregate multi-node RAM capacity.

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