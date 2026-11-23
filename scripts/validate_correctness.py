from pathlib import Path
import sys
import numpy as np
from dask.distributed import Client, LocalCluster

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.baselines import parallel_shared_memory, sequential_numpy, sequential_triple_loop
from src.distributed_matmul import run_distributed_matmul


def validate_all() -> None:
    data_dir = Path("data")
    a_small = np.load(data_dir / "A_250.npy")
    b_small = np.load(data_dir / "B_250.npy")

    ref_triple = sequential_triple_loop(a_small, b_small)
    ref_np_small = sequential_numpy(a_small, b_small)
    ref_par_small = parallel_shared_memory(a_small, b_small, num_workers=4)

    assert np.allclose(ref_triple, ref_np_small, rtol=1e-10, atol=1e-12), "Error en baseline NumPy"
    assert np.allclose(ref_triple, ref_par_small, rtol=1e-10, atol=1e-12), "Error en baseline paralelo"
    print("[PASS] Baselines validados frente a triple bucle en N=250 (rtol < 1e-10).")

    a_med = np.load(data_dir / "A_1000.npy")
    b_med = np.load(data_dir / "B_1000.npy")
    ref_np_med = sequential_numpy(a_med, b_med)

    cluster = LocalCluster(n_workers=4, threads_per_worker=1, processes=True, silence_logs=40)
    client = Client(cluster)

    try:
        for strategy in ("1d_row_broadcast", "2d_block"):
            c_dist, metrics = run_distributed_matmul(
                client=client,
                a=a_med,
                b=b_med,
                strategy=strategy,
                num_workers=4,
                threads_per_worker=1,
            )
            max_abs_err = float(np.max(np.abs(c_dist - ref_np_med)))
            assert np.allclose(c_dist, ref_np_med, rtol=1e-10, atol=1e-12), (
                f"Fallo de validacion en {strategy} (max_abs_err={max_abs_err})"
            )
            print(
                f"[PASS] Estrategia '{strategy}' validada en N=1000 | "
                f"error_max={max_abs_err:.2e} | tareas={metrics.total_tasks}"
            )
    finally:
        client.close()
        cluster.close()


if __name__ == "__main__":
    validate_all()