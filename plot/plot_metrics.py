"""Generate summary CSV files and scaling plots from Cholesky benchmark output."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# Apply the project plotting style
STYLE_FILE = Path(__file__).with_name("pythonStyle.mplstyle")
plt.style.use(STYLE_FILE)


class PerformancePlotter:
    """
    Create processed benchmark summaries and plots from one raw timing CSV.

    The input CSV is expected to contain the benchmark columns emitted by
    ``perf_scaling``:
    ``tag``, ``n``, ``repeat``, ``elapsed_seconds``, ``logdet``, and
    ``time_over_n3``.
    """

    def __init__(self, input_csv, output_dir):
        """
        Initialise the plotter, create the output directory, and load the CSV.

        Args:
            input_csv: Path to the raw benchmark CSV file.
            output_dir: Directory where processed CSV summaries and plots
                should be written.

        Raises:
            ValueError: If the CSV does not contain the required benchmark
                columns.
        """
        self.input_csv = Path(input_csv)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.df = pd.read_csv(self.input_csv)
        self._validate_columns()

    def _validate_columns(self):
        """
        Check that the loaded CSV contains all required benchmark columns.

        Raises:
            ValueError: If one or more expected columns are missing.
        """
        required_columns = {
            "tag",
            "n",
            "repeat",
            "elapsed_seconds",
            "logdet",
            "time_over_n3",
        }

        missing_columns = required_columns.difference(self.df.columns)
        if missing_columns:
            missing_str = ", ".join(sorted(missing_columns))
            raise ValueError(f"Missing required CSV columns: {missing_str}")

    def export_summary_csv(self):
        """
        Export a processed summary CSV grouped by matrix size.

        The summary contains aggregate timing statistics for each matrix size.
        """
        summary = self.df.groupby("n", as_index=False).agg(
            elapsed_median=("elapsed_seconds", "median"),
            elapsed_mean=("elapsed_seconds", "mean"),
            elapsed_std=("elapsed_seconds", "std"),
            elapsed_min=("elapsed_seconds", "min"),
            logdet_mean=("logdet", "mean"),
            logdet_std=("logdet", "std"),
        )

        summary.to_csv(self.output_dir / "summary_by_n.csv", index=False)

    def runtime_vs_n(self):
        """
        Plot median runtime against matrix size on log-log axes.

        The error bars show one standard deviation across the repeated runs for
        each matrix size.
        """
        summary = self.df.groupby("n", as_index=False).agg(
            elapsed_median=("elapsed_seconds", "median"),
            elapsed_std=("elapsed_seconds", "std"),
        )

        plt.figure(figsize=(7, 5))
        plt.errorbar(
            summary["n"],
            summary["elapsed_median"],
            yerr=summary["elapsed_std"].fillna(0.0),
            marker="s",
            capsize=4,
            label="Median runtime ± 1 std. dev.",
        )
        plt.xscale("log")
        plt.yscale("log")

        plt.xlabel("Matrix size n")
        plt.ylabel("Median runtime (s)")
        plt.title("Runtime scaling with matrix size")

        ax = plt.gca()
        # ax.set_xticks([1e2, 1e3])
        # ax.set_xticklabels([r"$10^2$", r"$10^3$"])
        # ax.set_xlim(1, 1200)

        plt.grid(True, which="major", axis="both")
        plt.minorticks_off()
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.output_dir / "runtime_vs_n.png", dpi=200)
        plt.close()

    def cubic_scaling_check(self):
        """Plot normalised runtime growth against a cubic reference on log-log axes."""
        summary = self.df.groupby("n", as_index=False).agg(
            elapsed_median=("elapsed_seconds", "median"),
            elapsed_std=("elapsed_seconds", "std"),
        )

        reference_n = summary["n"].iloc[0]
        reference_time = summary["elapsed_median"].iloc[0]
        normalised_runtime = summary["elapsed_median"] / reference_time
        normalised_std = (
            normalised_runtime
            * summary["elapsed_std"].fillna(0.0)
            / summary["elapsed_median"]
        )

        plt.figure(figsize=(7, 5))
        plt.errorbar(
            summary["n"] / reference_n,
            normalised_runtime,
            yerr=normalised_std.fillna(0.0),
            marker="s",
            capsize=4,
            label="Median runtime ± 1 std. dev.",
        )
        x_values = (summary["n"] / reference_n).tolist()
        y_values = [x * x * x for x in x_values]
        plt.plot(
            x_values,
            y_values,
            linestyle="--",
            color="black",
            linewidth=1.2,
            label=r"$n^3$ reference",
        )
        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel(r"Normalised matrix size $n / n_0$")
        plt.ylabel(r"Normalised runtime $T(n) / T(n_0)$")
        plt.title("Cubic scaling check on log-log axes")
        plt.grid(True, which="major", axis="both")
        plt.minorticks_off()
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.output_dir / "cubic_scaling_check.png", dpi=200)
        plt.close()

    def plot_all(self):
        """Run the full plotting pipeline for one benchmark CSV.

        This writes ``summary_by_n.csv`` plus the main runtime and scaling
        figures into :attr:`output_dir`.
        """
        legacy_plot = self.output_dir / "big_o_check.png"
        if legacy_plot.exists():
            legacy_plot.unlink()

        self.export_summary_csv()
        self.runtime_vs_n()
        self.cubic_scaling_check()


def main():
    """Run the plotting pipeline from the command line.

    Expected usage:
        ``python plot/plot_metrics.py <input_csv> <output_dir>``
    """
    if len(sys.argv) != 3:
        raise SystemExit(
            "Usage: python plot/plot_metrics.py <input_csv> <output_dir>"
        )

    input_csv = sys.argv[1]
    output_dir = sys.argv[2]

    plotter = PerformancePlotter(input_csv, output_dir)
    plotter.plot_all()


if __name__ == "__main__":
    main()
