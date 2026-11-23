import argparse
from pathlib import Path
import numpy as np


def generate_matrix_pair(n: int, seed: int, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    path_a = output_dir / f"A_{n}.npy"
    path_b = output_dir / f"B_{n}.npy"

    if path_a.exists() and path_b.exists():
        print(f"[SKIP] Matrices for N={n} already exist in {output_dir}")
        return

    rng = np.random.default_rng(seed)
    matrix_a = rng.standard_normal((n, n), dtype=np.float64)
    matrix_b = rng.standard_normal((n, n), dtype=np.float64)

    np.save(path_a, matrix_a)
    np.save(path_b, matrix_b)

    size_mb = (matrix_a.nbytes + matrix_b.nbytes) / (1024 ** 2)
    print(f"[OK] Generated N={n} ({size_mb:.2f} MB total) -> {path_a.name}, {path_b.name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate deterministic dense float64 matrices.")
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=[250, 500, 1000, 1500, 1890, 2000, 2381, 2726, 4000],
        help="Matrix dimensions N for N x N matrices.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--out", type=Path, default=Path("data"), help="Output directory.")
    args = parser.parse_args()

    for n in args.sizes:
        generate_matrix_pair(n=n, seed=args.seed + n, output_dir=args.out)


if __name__ == "__main__":
    main()