from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
    "figure.dpi": 300,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
})


def plot_paradigm_comparison(df: pd.DataFrame, out_dir: Path) -> None:
    paradigms = [
        ("sequential_numpy_blas", "Sequential (NumPy 1-thread BLAS)", "#2b5c8f"),
        ("parallel_shared_memory", "Shared-Memory Parallel (6 threads)", "#388e3c"),
        ("distributed_dask_1d_row_broadcast", "Distributed Dask 1D-Row (6 workers)", "#f57c00"),
        ("distributed_dask_2d_block", "Distributed Dask 2D-Block (6 workers)", "#d32f2f"),
    ]
    paradigm_keys = [p[0] for p in paradigms]
    sizes = [500, 1000, 2000]

    common = df[
        df["matrix_size"].isin(sizes) & df["paradigm"].isin(paradigm_keys)
    ].copy()

    max_y = float((common["median_s"] + common["iqr_s"]).max())
    x = np.arange(len(sizes))
    width = 0.19

    fig, ax = plt.subplots(figsize=(9, 4.8))
    for idx, (key, label, color) in enumerate(paradigms):
        sub = common[common["paradigm"] == key].set_index("matrix_size")
        medians = [sub.loc[n, "median_s"] for n in sizes]
        iqrs = [sub.loc[n, "iqr_s"] for n in sizes]
        bars = ax.bar(
            x + (idx - 1.5) * width,
            medians,
            width,
            yerr=iqrs,
            capsize=3,
            label=label,
            color=color,
            edgecolor="black",
            linewidth=0.6,
        )
        for bar, val, err in zip(bars, medians, iqrs):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                val + err + max_y * 0.02,
                f"{val:.3f}s",
                ha="center",
                va="bottom",
                fontsize=7.5,
            )

    ax.set_xticks(x)
    ax.set_xticklabels([f"N = {n}" for n in sizes])
    ax.set_ylabel("Median Wall-Clock Time (s)")
    ax.set_title("Execution Paradigm Comparison Across Common Matrix Sizes (Median ± IQR)")
    ax.set_ylim(0, max_y * 1.18)
    ax.legend(frameon=True, facecolor="white", framealpha=0.9)

    fig.tight_layout()
    fig.savefig(out_dir / "paradigm_comparison.png", dpi=300)
    plt.close(fig)


def plot_strong_scaling(df: pd.DataFrame, out_dir: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    workers = sorted(df["num_workers"].unique())

    styles = {
        "1d_row_broadcast": ("1D Row Broadcast", "#f57c00", "o", "-"),
        "2d_block": ("2D Block Partitioning", "#d32f2f", "s", "-"),
    }

    ax1.plot(workers, workers, linestyle=":", color="gray", label="Ideal Linear Speedup")
    ax2.axhline(1.0, linestyle=":", color="gray", label="Ideal Efficiency (100%)")

    for strat, (label, color, marker, ls) in styles.items():
        sub = df[df["strategy"] == strat].sort_values("num_workers")
        ax1.plot(
            sub["num_workers"],
            sub["speedup_vs_p1"],
            marker=marker,
            linestyle=ls,
            color=color,
            linewidth=2,
            label=label,
        )
        ax2.plot(
            sub["num_workers"],
            sub["efficiency_vs_p1"] * 100.0,
            marker=marker,
            linestyle=ls,
            color=color,
            linewidth=2,
            label=label,
        )

    ax1.set_title("Strong Scaling Speedup (N = 4000)")
    ax1.set_xlabel("Number of Worker Processes (P)")
    ax1.set_ylabel("Speedup vs. P=1 ($T_1 / T_P$)")
    ax1.set_xticks(workers)
    ax1.legend()

    ax2.set_title("Strong Scaling Parallel Efficiency (N = 4000)")
    ax2.set_xlabel("Number of Worker Processes (P)")
    ax2.set_ylabel("Parallel Efficiency (%)")
    ax2.set_xticks(workers)
    ax2.set_ylim(0, 115)
    ax2.legend()

    fig.tight_layout()
    fig.savefig(out_dir / "strong_scaling_speedup_efficiency.png", dpi=300)
    plt.close(fig)


def plot_weak_scaling(df: pd.DataFrame, out_dir: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    workers = sorted(df["num_workers"].unique())
    sizes_map = df.drop_duplicates("num_workers").set_index("num_workers")["matrix_size"].to_dict()
    x_labels = [f"P={p}\n(N={sizes_map[p]})" for p in workers]
    x = np.arange(len(workers))

    styles = {
        "1d_row_broadcast": ("1D Row Broadcast", "#f57c00", "o"),
        "2d_block": ("2D Block Partitioning", "#d32f2f", "s"),
    }

    for strat, (label, color, marker) in styles.items():
        sub = df[df["strategy"] == strat].sort_values("num_workers")
        ax1.errorbar(
            x,
            sub["median_s"],
            yerr=sub["iqr_s"],
            marker=marker,
            color=color,
            linewidth=2,
            capsize=4,
            label=label,
        )
        ax2.plot(
            x,
            sub["efficiency_vs_p1"] * 100.0,
            marker=marker,
            color=color,
            linewidth=2,
            label=label,
        )

    ax2.axhline(100.0, linestyle=":", color="gray", label="Ideal Weak Scaling (100%)")

    ax1.set_title("Weak Scaling Wall-Clock Time (6.75 GFLOP/worker)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(x_labels)
    ax1.set_ylabel("Median Total Time (s)")
    ax1.legend()

    ax2.set_title("Weak Scaling Efficiency ($T_1 / T_P$)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(x_labels)
    ax2.set_ylabel("Weak Scaling Efficiency (%)")
    ax2.set_ylim(0, 115)
    ax2.legend()

    fig.tight_layout()
    fig.savefig(out_dir / "weak_scaling_performance.png", dpi=300)
    plt.close(fig)


def plot_overhead_and_overlap(df: pd.DataFrame, out_dir: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.6))
    workers = sorted(df["num_workers"].unique())
    x = np.arange(len(workers))
    width = 0.34

    for idx, (strat, title_short) in enumerate([
        ("1d_row_broadcast", "1D Row"),
        ("2d_block", "2D Block"),
    ]):
        sub = df[df["strategy"] == strat].sort_values("num_workers")
        offset = -width / 2 if idx == 0 else width / 2
        setup = sub["setup_scatter_median_s"].values
        comp = sub["compute_e2e_median_s"].values

        ax1.bar(
            x + offset,
            comp,
            width,
            label=f"{title_short}: Compute + Gather E2E",
            color="#f57c00" if idx == 0 else "#d32f2f",
            edgecolor="black",
            linewidth=0.6,
        )
        ax1.bar(
            x + offset,
            setup,
            width,
            bottom=comp,
            label=f"{title_short}: Graph & Scatter Setup",
            color="#ffd54f" if idx == 0 else "#ef9a9a",
            edgecolor="black",
            linewidth=0.6,
        )

    ax1.set_title("End-to-End Wall-Clock Boundary Breakdown (N = 4000)")
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"P = {p}" for p in workers])
    ax1.set_ylabel("Wall-Clock Time (s)")
    ax1.legend(fontsize=8)

    for strat, label, color, marker in [
        ("1d_row_broadcast", "1D Row: Cumulative Task Transfer (Overlapped)", "#f57c00", "o"),
        ("2d_block", "2D Block: Cumulative Task Transfer (Overlapped)", "#d32f2f", "s"),
    ]:
        sub = df[df["strategy"] == strat].sort_values("num_workers")
        ax2.plot(
            workers,
            sub["task_transfer_sum_median_s"],
            marker=marker,
            color=color,
            linewidth=2,
            label=label,
        )

    sub_2d = df[df["strategy"] == "2d_block"].sort_values("num_workers")
    ax2.plot(
        workers,
        sub_2d["compute_e2e_median_s"],
        linestyle="--",
        color="#333333",
        marker="^",
        label="2D Block: Actual Compute+Gather Wall Time",
    )

    ax2.set_title("Overlapping Transfer Activity vs. Wall-Clock Time (N = 4000)")
    ax2.set_xlabel("Number of Worker Processes (P)")
    ax2.set_ylabel("Time (s)")
    ax2.set_xticks(workers)
    ax2.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out_dir / "overhead_and_overlap_breakdown.png", dpi=300)
    plt.close(fig)


def main() -> None:
    results_dir = Path("results")
    plots_dir = results_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    df_paradigm = pd.read_csv(results_dir / "paradigm_comparison.csv")
    df_strong = pd.read_csv(results_dir / "strong_scaling.csv")
    df_weak = pd.read_csv(results_dir / "weak_scaling.csv")

    plot_paradigm_comparison(df_paradigm, plots_dir)
    plot_strong_scaling(df_strong, plots_dir)
    plot_weak_scaling(df_weak, plots_dir)
    plot_overhead_and_overlap(df_strong, plots_dir)

    print(f"[OK] All 4 publication plots saved to {plots_dir}")


if __name__ == "__main__":
    main()