import argparse
import os
from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd
from dask.distributed import Client, LocalCluster

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.baselines import parallel_shared_memory, sequential_numpy, sequential_triple_loop
from src.distributed_matmul import run_distributed_matmul


def summarize_times(times: list[float]) -> dict[str, float]:
    arr = np.array(times, dtype=np.float64)
    q75, q25 = np.percentile(arr, [75, 25])
    return {
        "median_s": float(np.median(arr)),
        "mean_s": float(np.mean(arr)),
        "std_s": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
        "iqr_s": float(q75 - q25),
        "min_s": float(np.min(arr)),
        "max_s": float(np.max(arr)),
    }


def benchmark_baselines(data_dir: Path, results_dir: Path, runs: int, max_workers: int) -> None:
    rows = []

    for n in (250, 500):
        a = np.load(data_dir / f"A_{n}.npy")
        b = np.load(data_dir / f"B_{n}.npy")
        times = []
        for _ in range(max(1, runs - 2)):
            t0 = time.perf_counter()
            _ = sequential_triple_loop(a, b)
            times.append(time.perf_counter() - t0)
        stats = summarize_times(times)
        rows.append({
            "paradigm": "sequential_triple_loop_py",
            "matrix_size": n,
            "workers": 1,
            "threads_per_worker": 1,
            "setup_s": 0.0,
            "compute_e2e_s": stats["median_s"],
            **stats,
        })
        print(f"[Baseline Triple-Loop] N={n} -> median={stats['median_s']:.4f}s")

    for n in (500, 1000, 2000):
        a = np.load(data_dir / f"A_{n}.npy")
        b = np.load(data_dir / f"B_{n}.npy")

        _ = sequential_numpy(a, b)

        seq_times = []
        for _ in range(runs):
            t0 = time.perf_counter()
            _ = sequential_numpy(a, b)
            seq_times.append(time.perf_counter() - t0)
        seq_stats = summarize_times(seq_times)
        rows.append({
            "paradigm": "sequential_numpy_blas",
            "matrix_size": n,
            "workers": 1,
            "threads_per_worker": 1,
            "setup_s": 0.0,
            "compute_e2e_s": seq_stats["median_s"],
            **seq_stats,
        })

        par_times = []
        for _ in range(runs):
            t0 = time.perf_counter()
            _ = parallel_shared_memory(a, b, num_workers=max_workers)
            par_times.append(time.perf_counter() - t0)
        par_stats = summarize_times(par_times)
        rows.append({
            "paradigm": "parallel_shared_memory",
            "matrix_size": n,
            "workers": max_workers,
            "threads_per_worker": 1,
            "setup_s": 0.0,
            "compute_e2e_s": par_stats["median_s"],
            **par_stats,
        })
        print(
            f"[Baselines N={n}] SeqNumPy={seq_stats['median_s']:.4f}s | "
            f"ParShared({max_workers})={par_stats['median_s']:.4f}s"
        )

    cluster = LocalCluster(
        n_workers=max_workers,
        threads_per_worker=1,
        processes=True,
        silence_logs=40,
    )
    client = Client(cluster)
    try:
        for n in (500, 1000, 2000):
            a = np.load(data_dir / f"A_{n}.npy")
            b = np.load(data_dir / f"B_{n}.npy")
            for strategy in ("1d_row_broadcast", "2d_block"):
                _, _ = run_distributed_matmul(
                    client, a, b, strategy=strategy, num_workers=max_workers
                )
                wall_times = []
                setup_times = []
                comp_times = []
                for _ in range(runs):
                    _, m = run_distributed_matmul(
                        client, a, b, strategy=strategy, num_workers=max_workers
                    )
                    wall_times.append(m.total_wall_s)
                    setup_times.append(m.setup_scatter_s)
                    comp_times.append(m.compute_e2e_s)

                stats = summarize_times(wall_times)
                rows.append({
                    "paradigm": f"distributed_dask_{strategy}",
                    "matrix_size": n,
                    "workers": max_workers,
                    "threads_per_worker": 1,
                    "setup_s": float(np.median(setup_times)),
                    "compute_e2e_s": float(np.median(comp_times)),
                    **stats,
                })
                print(
                    f"[Distributed {strategy} N={n}] P={max_workers} -> "
                    f"total_median={stats['median_s']:.4f}s (compute={np.median(comp_times):.4f}s)"
                )
    finally:
        client.close()
        cluster.close()

    df = pd.DataFrame(rows)
    out_path = results_dir / "paradigm_comparison.csv"
    df.to_csv(out_path, index=False)
    print(f"[SAVED] {out_path}")


def benchmark_scaling(
    data_dir: Path,
    results_dir: Path,
    worker_counts: list[int],
    runs: int,
    strong_n: int = 4000,
    weak_base_n: int = 1500,
) -> None:
    strong_rows = []
    weak_rows = []

    a_strong = np.load(data_dir / f"A_{strong_n}.npy")
    b_strong = np.load(data_dir / f"B_{strong_n}.npy")

    for p in worker_counts:
        cluster = LocalCluster(
            n_workers=p,
            threads_per_worker=1,
            processes=True,
            silence_logs=40,
        )
        client = Client(cluster)
        try:
            for strategy in ("1d_row_broadcast", "2d_block"):
                _, _ = run_distributed_matmul(
                    client, a_strong, b_strong, strategy=strategy, num_workers=p
                )
                run_metrics = []
                for r_idx in range(runs):
                    rep_file = None
                    if p == max(worker_counts) and r_idx == 0:
                        rep_file = str(
                            results_dir / "reports" / f"strong_N{strong_n}_P{p}_{strategy}.html"
                        )
                    _, m = run_distributed_matmul(
                        client,
                        a_strong,
                        b_strong,
                        strategy=strategy,
                        num_workers=p,
                        report_path=rep_file,
                    )
                    run_metrics.append(m.to_dict())

                df_runs = pd.DataFrame(run_metrics)
                stats = summarize_times(df_runs["total_wall_s"].tolist())
                strong_rows.append({
                    "experiment": "strong_scaling",
                    "environment": "local_multiprocess_1node",
                    "strategy": strategy,
                    "matrix_size": strong_n,
                    "num_workers": p,
                    "threads_per_worker": 1,
                    "num_chunks_a": int(df_runs["num_chunks_a"].iloc[0]),
                    "num_chunks_b": int(df_runs["num_chunks_b"].iloc[0]),
                    "total_tasks": int(df_runs["total_tasks"].iloc[0]),
                    "setup_scatter_median_s": float(df_runs["setup_scatter_s"].median()),
                    "compute_e2e_median_s": float(df_runs["compute_e2e_s"].median()),
                    "task_compute_sum_median_s": float(df_runs["task_compute_sum_s"].median()),
                    "task_transfer_sum_median_s": float(df_runs["task_transfer_sum_s"].median()),
                    "task_serialize_sum_median_s": float(df_runs["task_serialize_sum_s"].median()),
                    "peak_worker_memory_mb": float(df_runs["peak_worker_memory_mb"].max()),
                    "total_transferred_mb": float(df_runs["total_transferred_mb"].median()),
                    **stats,
                })
                print(
                    f"[Strong N={strong_n}] P={p} | {strategy} -> "
                    f"total={stats['median_s']:.4f}s (compute={df_runs['compute_e2e_s'].median():.4f}s, "
                    f"transfer_sum={df_runs['task_transfer_sum_s'].median():.4f}s)"
                )

            n_weak = int(round(weak_base_n * (p ** (1.0 / 3.0))))
            a_weak = np.load(data_dir / f"A_{n_weak}.npy")
            b_weak = np.load(data_dir / f"B_{n_weak}.npy")

            for strategy in ("1d_row_broadcast", "2d_block"):
                _, _ = run_distributed_matmul(
                    client, a_weak, b_weak, strategy=strategy, num_workers=p
                )
                run_metrics = []
                for _ in range(runs):
                    _, m = run_distributed_matmul(
                        client, a_weak, b_weak, strategy=strategy, num_workers=p
                    )
                    run_metrics.append(m.to_dict())

                df_runs = pd.DataFrame(run_metrics)
                stats = summarize_times(df_runs["total_wall_s"].tolist())
                work_per_worker_gflops = (2.0 * (n_weak ** 3)) / (p * 1e9)
                weak_rows.append({
                    "experiment": "weak_scaling",
                    "environment": "local_multiprocess_1node",
                    "strategy": strategy,
                    "matrix_size": n_weak,
                    "num_workers": p,
                    "threads_per_worker": 1,
                    "work_per_worker_gflops": round(work_per_worker_gflops, 3),
                    "total_tasks": int(df_runs["total_tasks"].iloc[0]),
                    "setup_scatter_median_s": float(df_runs["setup_scatter_s"].median()),
                    "compute_e2e_median_s": float(df_runs["compute_e2e_s"].median()),
                    "task_compute_sum_median_s": float(df_runs["task_compute_sum_s"].median()),
                    "task_transfer_sum_median_s": float(df_runs["task_transfer_sum_s"].median()),
                    "peak_worker_memory_mb": float(df_runs["peak_worker_memory_mb"].max()),
                    "total_transferred_mb": float(df_runs["total_transferred_mb"].median()),
                    **stats,
                })
                print(
                    f"[Weak N={n_weak}] P={p} ({work_per_worker_gflops:.2f} GFLOP/worker) | "
                    f"{strategy} -> total={stats['median_s']:.4f}s"
                )
        finally:
            client.close()
            cluster.close()

    df_strong = pd.DataFrame(strong_rows)
    df_weak = pd.DataFrame(weak_rows)

    for df in (df_strong, df_weak):
        base_map = (
            df[df["num_workers"] == 1]
            .set_index("strategy")["median_s"]
            .to_dict()
        )
        df["speedup_vs_p1"] = df.apply(
            lambda r: base_map[r["strategy"]] / r["median_s"], axis=1
        )
        df["efficiency_vs_p1"] = df.apply(
            lambda r: (base_map[r["strategy"]] / r["median_s"]) / r["num_workers"], axis=1
        )

    strong_path = results_dir / "strong_scaling.csv"
    weak_path = results_dir / "weak_scaling.csv"
    df_strong.to_csv(strong_path, index=False)
    df_weak.to_csv(weak_path, index=False)
    print(f"[SAVED] {strong_path}")
    print(f"[SAVED] {weak_path}")


def benchmark_remote_cluster(
    scheduler_address: str,
    data_dir: Path,
    results_dir: Path,
    runs: int,
    label: str,
) -> None:
    client = Client(scheduler_address)
    try:
        info = client.scheduler_info()
        workers = info.get("workers", {})
        num_workers = len(workers)
        hosts = sorted({w.get("host", "unknown") for w in workers.values()})
        print(
            f"[Remote Cluster '{label}'] Connected to {scheduler_address} | "
            f"workers={num_workers} | physical_hosts={len(hosts)} ({hosts})"
        )

        rows = []
        for n in (2000, 4000):
            a = np.load(data_dir / f"A_{n}.npy")
            b = np.load(data_dir / f"B_{n}.npy")
            for strategy in ("1d_row_broadcast", "2d_block"):
                _, _ = run_distributed_matmul(
                    client, a, b, strategy=strategy, num_workers=num_workers
                )
                run_metrics = []
                for _ in range(runs):
                    _, m = run_distributed_matmul(
                        client, a, b, strategy=strategy, num_workers=num_workers
                    )
                    run_metrics.append(m.to_dict())

                df_runs = pd.DataFrame(run_metrics)
                stats = summarize_times(df_runs["total_wall_s"].tolist())
                rows.append({
                    "environment": label,
                    "physical_hosts": len(hosts),
                    "num_workers": num_workers,
                    "strategy": strategy,
                    "matrix_size": n,
                    "setup_scatter_median_s": float(df_runs["setup_scatter_s"].median()),
                    "compute_e2e_median_s": float(df_runs["compute_e2e_s"].median()),
                    "task_compute_sum_median_s": float(df_runs["task_compute_sum_s"].median()),
                    "task_transfer_sum_median_s": float(df_runs["task_transfer_sum_s"].median()),
                    "peak_worker_memory_mb": float(df_runs["peak_worker_memory_mb"].max()),
                    "total_transferred_mb": float(df_runs["total_transferred_mb"].median()),
                    **stats,
                })
                print(
                    f"[{label} N={n}] {strategy} -> total={stats['median_s']:.4f}s "
                    f"(scatter={df_runs['setup_scatter_s'].median():.4f}s, "
                    f"compute={df_runs['compute_e2e_s'].median():.4f}s)"
                )

        out_path = results_dir / f"cluster_{label}.csv"
        pd.DataFrame(rows).to_csv(out_path, index=False)
        print(f"[SAVED] {out_path}")
    finally:
        client.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run distributed matrix multiplication benchmarks.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--runs", type=int, default=5, help="Timed repetitions per configuration.")
    parser.add_argument("--workers", type=int, nargs="+", default=[1, 2, 4, 6])
    parser.add_argument(
        "--scheduler-address",
        type=str,
        default=None,
        help="Optional tcp://IP:8786 address for multi-node cluster benchmarks.",
    )
    parser.add_argument(
        "--cluster-label",
        type=str,
        default="2nodes_lan",
        help="Label for remote cluster output CSV.",
    )
    parser.add_argument(
        "--skip-baselines",
        action="store_true",
        help="Skip paradigm_comparison.csv if already generated.",
    )
    args = parser.parse_args()

    args.results_dir.mkdir(parents=True, exist_ok=True)
    (args.results_dir / "reports").mkdir(parents=True, exist_ok=True)

    if args.scheduler_address:
        benchmark_remote_cluster(
            scheduler_address=args.scheduler_address,
            data_dir=args.data_dir,
            results_dir=args.results_dir,
            runs=args.runs,
            label=args.cluster_label,
        )
    else:
        if not args.skip_baselines:
            max_w = max(args.workers)
            benchmark_baselines(
                data_dir=args.data_dir,
                results_dir=args.results_dir,
                runs=args.runs,
                max_workers=max_w,
            )
        benchmark_scaling(
            data_dir=args.data_dir,
            results_dir=args.results_dir,
            worker_counts=args.workers,
            runs=args.runs,
        )


if __name__ == "__main__":
    main()