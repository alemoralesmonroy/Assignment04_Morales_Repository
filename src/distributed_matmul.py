from contextlib import nullcontext
from dataclasses import dataclass, asdict
import time
from typing import Literal, Optional
import numpy as np
import dask.array as da
from dask.distributed import Client, get_task_stream, performance_report


PartitionStrategy = Literal["1d_row_broadcast", "2d_block"]


@dataclass
class DistributedMetrics:
    strategy: str
    matrix_size: int
    num_workers: int
    threads_per_worker: int
    num_chunks_a: int
    num_chunks_b: int
    total_tasks: int
    setup_scatter_s: float
    compute_e2e_s: float
    total_wall_s: float
    task_compute_sum_s: float
    task_transfer_sum_s: float
    task_serialize_sum_s: float
    peak_worker_memory_mb: float
    total_transferred_mb: float

    def to_dict(self) -> dict:
        return asdict(self)


def run_distributed_matmul(
    client: Client,
    a: np.ndarray,
    b: np.ndarray,
    strategy: PartitionStrategy = "1d_row_broadcast",
    num_workers: int = 1,
    threads_per_worker: int = 1,
    block_divisor: Optional[int] = None,
    report_path: Optional[str] = None,
) -> tuple[np.ndarray, DistributedMetrics]:
    n = a.shape[0]
    divisions = block_divisor if block_divisor is not None else max(1, num_workers)

    t0_setup = time.perf_counter()
    if strategy == "1d_row_broadcast":
        chunk_rows = max(1, int(np.ceil(n / divisions)))
        da_a = da.from_array(a, chunks=(chunk_rows, n))
        da_b = da.from_array(b, chunks=(n, n))
    elif strategy == "2d_block":
        chunk_dim = max(1, int(np.ceil(n / divisions)))
        da_a = da.from_array(a, chunks=(chunk_dim, chunk_dim))
        da_b = da.from_array(b, chunks=(chunk_dim, chunk_dim))
    else:
        raise ValueError(f"Unsupported partitioning strategy: {strategy}")

    da_c = da.matmul(da_a, da_b)
    total_tasks = len(da_c.__dask_graph__())
    num_chunks_a = da_a.npartitions
    num_chunks_b = da_b.npartitions
    t1_setup = time.perf_counter()

    perf_ctx = performance_report(filename=report_path) if report_path else nullcontext()

    t0_compute = time.perf_counter()
    with perf_ctx, get_task_stream(plot=False) as ts:
        result = da_c.compute()
    t1_compute = time.perf_counter()

    task_compute_s = 0.0
    task_transfer_s = 0.0
    task_serialize_s = 0.0
    transferred_bytes = 0

    for entry in ts.data:
        for start_stop in entry.get("startstops", []):
            action = start_stop.get("action", "")
            duration = max(0.0, start_stop.get("stop", 0.0) - start_stop.get("start", 0.0))
            if action == "compute":
                task_compute_s += duration
            elif action == "transfer":
                task_transfer_s += duration
                transferred_bytes += int(entry.get("nbytes", 0))
            elif action in ("disk-read", "disk-write", "serialize", "deserialize"):
                task_serialize_s += duration

    scheduler_info = client.scheduler_info()
    workers_info = scheduler_info.get("workers", {})
    peak_mem_bytes = 0
    for w_meta in workers_info.values():
        mem_used = w_meta.get("metrics", {}).get("memory", 0)
        if mem_used > peak_mem_bytes:
            peak_mem_bytes = mem_used

    setup_s = t1_setup - t0_setup
    compute_s = t1_compute - t0_compute

    metrics = DistributedMetrics(
        strategy=strategy,
        matrix_size=n,
        num_workers=num_workers,
        threads_per_worker=threads_per_worker,
        num_chunks_a=num_chunks_a,
        num_chunks_b=num_chunks_b,
        total_tasks=total_tasks,
        setup_scatter_s=setup_s,
        compute_e2e_s=compute_s,
        total_wall_s=setup_s + compute_s,
        task_compute_sum_s=task_compute_s,
        task_transfer_sum_s=task_transfer_s,
        task_serialize_sum_s=task_serialize_s,
        peak_worker_memory_mb=peak_mem_bytes / (1024 ** 2),
        total_transferred_mb=transferred_bytes / (1024 ** 2),
    )

    return result, metrics