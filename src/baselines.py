from concurrent.futures import ThreadPoolExecutor
import numpy as np


def sequential_triple_loop(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    n = a.shape[0]
    c = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for k in range(n):
            a_ik = a[i, k]
            for j in range(n):
                c[i, j] += a_ik * b[k, j]
    return c


def sequential_numpy(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.matmul(a, b)


def parallel_shared_memory(a: np.ndarray, b: np.ndarray, num_workers: int = 6) -> np.ndarray:
    n = a.shape[0]
    c = np.empty((n, n), dtype=np.float64)
    row_splits = np.array_split(np.arange(n), num_workers)

    def _compute_row_block(rows: np.ndarray) -> None:
        if rows.size > 0:
            c[rows[0] : rows[-1] + 1, :] = np.matmul(a[rows[0] : rows[-1] + 1, :], b)

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        list(executor.map(_compute_row_block, row_splits))

    return c