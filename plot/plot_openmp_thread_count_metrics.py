"""
Generate summary CSV and thread-scaling plots from OpenMP thread-count sweep output.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

STYLE_FILE = Path(__file__).with_name("pythonStyle.mplstyle")
plt.style.use(STYLE_FILE)


class OpenMPThreadCountPlotter:
    """Create summary data and plots from one OpenMP thread-count sweep CSV."""

    def __init__(self, input_csv, output_dir):
        self.input_csv = Path(input_csv)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.df = pd.read_csv(self.input_csv)
        self._validate_columns()

    def _validate_columns(self):
        required_columns = {
            "tag",
            "n",
            "threads",
            "repeat",
            "elapsed_seconds",
            "logdet",
            "time_over_n3",
        }

        missing_columns = required_columns.difference(self.df.columns)
        if missing_columns:
            missing_str = ", ".join(sorted(missing_columns))
            raise ValueError("Missing required CSV columns: " + missing_str)

    def export_summary_csv(self):
        summary = self.df.groupby(["tag", "n", "threads"], as_index=False).agg(
            elapsed_median=("elapsed_seconds", "median"),
            elapsed_mean=("elapsed_seconds", "mean"),
            elapsed_std=("elapsed_seconds", "std"),
            elapsed_min=("elapsed_seconds", "min"),
            logdet_mean=("logdet", "mean"),
            logdet_std=("logdet", "std"),
        )

        openmp_summary = summary[summary["tag"] != "baseline"].copy()

        openmp_rows = self.df[self.df["tag"] != "baseline"].copy()
        single_thread_rows = openmp_rows.loc[
            openmp_rows["threads"] == 1, ["tag", "n", "repeat", "elapsed_seconds"]
        ].rename(columns={"elapsed_seconds": "elapsed_single_thread"})

        speedup_rows = openmp_rows.merge(
            single_thread_rows, on=["tag", "n", "repeat"], how="left"
        )
        if speedup_rows["elapsed_single_thread"].isna().any():
            raise ValueError("Each OpenMP method must include a 1-thread run for speedup calculation.")

        speedup_rows["speedup_vs_one_thread"] = (
            speedup_rows["elapsed_single_thread"] / speedup_rows["elapsed_seconds"]
        )

        speedup_summary = speedup_rows.groupby(["tag", "n", "threads"], as_index=False).agg(
            speedup_median=("speedup_vs_one_thread", "median"),
            speedup_mean=("speedup_vs_one_thread", "mean"),
            speedup_std=("speedup_vs_one_thread", "std"),
        )

        merged = openmp_summary.merge(speedup_summary, on=["tag", "n", "threads"], how="left")
        merged["parallel_efficiency"] = merged["speedup_median"] / merged["threads"]

        merged.to_csv(self.output_dir / "summary_by_threads.csv", index=False)
        return merged

    def speedup_vs_thread_count(self, summary):
        plt.figure(figsize=(7, 5))

        for tag, group in summary.groupby("tag"):
            ordered = group.sort_values("threads")
            plt.errorbar(
                ordered["threads"],
                ordered["speedup_median"],
                yerr=ordered["speedup_std"].fillna(0.0),
                marker="o",
                capsize=4,
                label=tag,
            )

        plt.axhline(1.0, color="black", linewidth=1.0, linestyle="--")
        plt.xlabel("Thread count")
        plt.ylabel("Speedup vs 1 thread")
        plt.title("OpenMP speedup vs thread count")
        plt.grid(True, which="major", axis="both")
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.output_dir / "speedup_vs_thread_count.png", dpi=200)
        plt.close()

    def plot_all(self):
        summary = self.export_summary_csv()
        self.speedup_vs_thread_count(summary)


def main():
    if len(sys.argv) != 3:
        raise SystemExit(
            "Usage: python plot/plot_openmp_thread_count_metrics.py <input_csv> <output_dir>"
        )

    input_csv = sys.argv[1]
    output_dir = sys.argv[2]

    plotter = OpenMPThreadCountPlotter(input_csv, output_dir)
    plotter.plot_all()


if __name__ == "__main__":
    main()
