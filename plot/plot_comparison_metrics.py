"""Generate cross-implementation comparison plots from multiple benchmark CSV files.

This module combines the raw CSV files emitted by ``perf_scaling`` for several
optimisations, writes processed summary tables, and creates report-ready plots
that compare runtime, speedup, and dense memory growth.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# Apply the shared project plotting style.
STYLE_FILE = Path(__file__).with_name("pythonStyle.mplstyle")
plt.style.use(STYLE_FILE)


class ComparisonPlotter:
    """Create cross-implementation summaries and plots from multiple raw CSV files."""

    def __init__(self, input_csvs: list[str | Path], output_dir: str | Path) -> None:
        """Load and validate several raw benchmark CSV files.

        Args:
            input_csvs: Paths to the raw CSV files produced by ``perf_scaling``.
            output_dir: Directory where summary CSV files and comparison plots
                should be written.
        """
        self.input_csvs = [Path(path) for path in input_csvs]
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.df = pd.concat((pd.read_csv(path) for path in self.input_csvs), ignore_index=True)
        self._validate_columns()
        self.summary = self._build_summary()

    def _validate_columns(self) -> None:
        """Check that the combined dataframe contains the expected benchmark columns."""
        required_columns = {
            "tag",
            "n",
            "repeat",
            "elapsed_seconds",
            "logdet",
            "matrix_bytes",
            "flop_estimate",
            "gflops_est",
            "time_over_n3",
        }

        missing_columns = required_columns.difference(self.df.columns)
        if missing_columns:
            missing_str = ", ".join(sorted(missing_columns))
            raise ValueError(f"Missing required CSV columns: {missing_str}")

    def _build_summary(self) -> pd.DataFrame:
        """Aggregate repeated runs by optimisation method and matrix size."""
        summary = self.df.groupby(["tag", "n"], as_index=False).agg(
            elapsed_median=("elapsed_seconds", "median"),
            elapsed_mean=("elapsed_seconds", "mean"),
            elapsed_std=("elapsed_seconds", "std"),
            elapsed_min=("elapsed_seconds", "min"),
            time_over_n3_median=("time_over_n3", "median"),
            time_over_n3_mean=("time_over_n3", "mean"),
            time_over_n3_std=("time_over_n3", "std"),
            logdet_mean=("logdet", "mean"),
            logdet_std=("logdet", "std"),
            matrix_bytes=("matrix_bytes", "first"),
            gflops_median=("gflops_est", "median"),
        )

        summary["matrix_mib"] = summary["matrix_bytes"] / (1024**2)
        summary["matrix_gib"] = summary["matrix_bytes"] / (1024**3)

        baseline = summary.loc[summary["tag"] == "baseline", ["n", "elapsed_median"]].rename(
            columns={"elapsed_median": "baseline_elapsed_median"}
        )

        merged = summary.merge(baseline, on="n", how="left")
        merged["speedup_vs_baseline"] = (
            merged["baseline_elapsed_median"] / merged["elapsed_median"]
        )
        return merged

    def _save_plot(self, filename: str) -> None:
        """Apply common formatting and save the current matplotlib figure."""
        plt.tight_layout()
        plt.savefig(self.output_dir / filename, dpi=200)
        plt.close()

    def export_summary_csv(self) -> None:
        """Write the processed summary table and combined raw CSV to disk."""
        self.df.to_csv(self.output_dir / "combined_raw.csv", index=False)
        self.summary.to_csv(self.output_dir / "summary_by_method_and_n.csv", index=False)

    def runtime_vs_n_by_method(self) -> None:
        """Plot median runtime against matrix size for each implementation."""
        plt.figure(figsize=(8, 5.5))

        for tag, group in self.summary.groupby("tag"):
            ordered = group.sort_values("n")
            plt.errorbar(
                ordered["n"],
                ordered["elapsed_median"],
                yerr=ordered["elapsed_std"].fillna(0.0),
                marker="o",
                capsize=3,
                label=tag,
            )

        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel("Matrix size n")
        plt.ylabel("Median runtime (s)")
        plt.title("Runtime scaling by implementation")
        plt.grid(True, which="major", axis="both")
        plt.minorticks_off()
        plt.legend()
        self._save_plot("runtime_vs_n_by_method.png")

    def speedup_vs_n(self) -> None:
        """Plot median speedup relative to the baseline implementation."""
        plt.figure(figsize=(8, 5.5))

        for tag, group in self.summary.groupby("tag"):
            ordered = group.sort_values("n")
            plt.plot(
                ordered["n"],
                ordered["speedup_vs_baseline"],
                marker="o",
                label=tag,
            )

        plt.xscale("log")
        plt.xlabel("Matrix size n")
        plt.ylabel("Speedup vs baseline")
        plt.title("Speedup relative to baseline")
        plt.grid(True, which="major", axis="both")
        plt.minorticks_off()
        plt.legend()
        self._save_plot("speedup_vs_baseline.png")

    def memory_growth(self) -> None:
        """Plot dense matrix storage growth as a function of matrix size."""
        memory = (
            self.summary.loc[:, ["n", "matrix_gib"]]
            .drop_duplicates()
            .sort_values("n")
        )

        plt.figure(figsize=(7, 5))
        plt.plot(memory["n"], memory["matrix_gib"], marker="o")
        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel("Matrix size n")
        plt.ylabel("Dense matrix storage (GiB)")
        plt.title("Dense storage growth with matrix size")
        plt.grid(True, which="major", axis="both")
        plt.minorticks_off()
        self._save_plot("memory_growth.png")

    def plot_all(self) -> None:
        """Run the full comparison plotting pipeline."""
        self.export_summary_csv()
        self.runtime_vs_n_by_method()
        self.speedup_vs_n()
        self.memory_growth()


def main() -> None:
    """Run the comparison plotting pipeline from the command line.

    Expected usage:
        ``python plot/plot_comparison_metrics.py <output_dir> <csv1> [csv2 ...]``
    """
    if len(sys.argv) < 3:
        raise SystemExit(
            "Usage: python plot/plot_comparison_metrics.py <output_dir> <csv1> [csv2 ...]"
        )

    output_dir = sys.argv[1]
    input_csvs = sys.argv[2:]

    plotter = ComparisonPlotter(input_csvs, output_dir)
    plotter.plot_all()


if __name__ == "__main__":
    main()
