"""
Generate summary CSV files and block-size plots from blocked Cholesky benchmark output.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

STYLE_FILE = Path(__file__).with_name("pythonStyle.mplstyle")
plt.style.use(STYLE_FILE)


class BlockSizePlotter:
    """Create summary tables and plots from one block-size sweep CSV."""

    def __init__(self, input_csv: str | Path, output_dir: str | Path):
        self.input_csv = Path(input_csv)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.df = pd.read_csv(self.input_csv)
        self._validate_columns()

    def _validate_columns(self) -> None:
        required_columns = {
            "optimisation",
            "n",
            "block_size",
            "repeat",
            "elapsed_seconds",
            "speedup_factor_vs_baseline",
            "logdet_library",
            "logdet_factor",
            "relative_difference_percent",
        }

        missing_columns = required_columns.difference(self.df.columns)
        if missing_columns:
            missing_str = ", ".join(sorted(missing_columns))
            raise ValueError(f"Missing required CSV columns: {missing_str}")

    def export_summary_csv(self) -> pd.DataFrame:
        summary = self.df.groupby(["optimisation", "block_size"], as_index=False).agg(
            elapsed_median=("elapsed_seconds", "median"),
            elapsed_mean=("elapsed_seconds", "mean"),
            elapsed_std=("elapsed_seconds", "std"),
            speedup_median=("speedup_factor_vs_baseline", "median"),
            speedup_mean=("speedup_factor_vs_baseline", "mean"),
            speedup_std=("speedup_factor_vs_baseline", "std"),
            relative_difference_median=("relative_difference_percent", "median"),
            relative_difference_mean=("relative_difference_percent", "mean"),
        )

        summary.to_csv(self.output_dir / "summary_by_block_size.csv", index=False)
        return summary

    def runtime_vs_block_size(self, summary: pd.DataFrame) -> None:
        plt.figure(figsize=(7, 5))

        for optimisation, group in summary.groupby("optimisation"):
            group = group.sort_values("block_size")
            plt.errorbar(
                group["block_size"],
                group["elapsed_median"],
                yerr=group["elapsed_std"].fillna(0.0),
                marker="o",
                capsize=4,
                label=optimisation,
            )

        plt.xlabel("Block size")
        plt.ylabel("Median runtime (s)")
        plt.title("Blocked Cholesky runtime vs block size")
        plt.grid(True, which="major", axis="both")
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.output_dir / "runtime_vs_block_size.png", dpi=200)
        plt.close()

    def speedup_vs_block_size(self, summary: pd.DataFrame) -> None:
        plt.figure(figsize=(7, 5))

        for optimisation, group in summary.groupby("optimisation"):
            group = group.sort_values("block_size")
            plt.errorbar(
                group["block_size"],
                group["speedup_median"],
                yerr=group["speedup_std"].fillna(0.0),
                marker="o",
                capsize=4,
                label=optimisation,
            )

        plt.xlabel("Block size")
        plt.ylabel("Median speedup vs baseline")
        plt.title("Blocked Cholesky speedup vs block size")
        plt.grid(True, which="major", axis="both")
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.output_dir / "speedup_vs_block_size.png", dpi=200)
        plt.close()

    def plot_all(self) -> None:
        summary = self.export_summary_csv()
        self.runtime_vs_block_size(summary)
        self.speedup_vs_block_size(summary)


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit(
            "Usage: python plot/plot_block_size_metrics.py <input_csv> <output_dir>"
        )

    input_csv = sys.argv[1]
    output_dir = sys.argv[2]

    plotter = BlockSizePlotter(input_csv, output_dir)
    plotter.plot_all()


if __name__ == "__main__":
    main()
