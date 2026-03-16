"""Generate cross-implementation comparison plots from one or more benchmark CSV files."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# Apply the shared project plotting style.
STYLE_FILE = Path(__file__).with_name("pythonStyle.mplstyle")
plt.style.use(STYLE_FILE)


class ComparisonPlotter:
    """Create cross-implementation plots from one or more raw CSV files."""

    def __init__(self, input_csvs, output_dir):
        """Load and validate several raw benchmark CSV files.

        Args:
            input_csvs: Paths to the raw CSV files produced by ``perf_scaling``.
            output_dir: Directory where comparison plots should be written.
        """
        self.input_csvs = [Path(path) for path in input_csvs]
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.df = pd.concat((pd.read_csv(path) for path in self.input_csvs), ignore_index=True)
        self._validate_columns()
        self.has_baseline = "baseline" in set(self.df["tag"])
        self.summary = self._build_summary()

    def _validate_columns(self):
        """Check that the combined dataframe contains the expected benchmark columns."""
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

    def _build_summary(self):
        """Aggregate repeated runs by optimisation method and matrix size."""
        summary = self.df.groupby(["tag", "n"], as_index=False).agg(
            elapsed_median=("elapsed_seconds", "median"),
            elapsed_mean=("elapsed_seconds", "mean"),
            elapsed_std=("elapsed_seconds", "std"),
            elapsed_min=("elapsed_seconds", "min"),
            logdet_mean=("logdet", "mean"),
            logdet_std=("logdet", "std"),
        )

        if not self.has_baseline:
            summary["speedup_median"] = float("nan")
            summary["speedup_mean"] = float("nan")
            summary["speedup_std"] = float("nan")
            return summary

        baseline_runs = self.df.loc[
            self.df["tag"] == "baseline", ["n", "repeat", "elapsed_seconds"]
        ].rename(columns={"elapsed_seconds": "baseline_elapsed_seconds"})

        speedup_rows = self.df.merge(baseline_runs, on=["n", "repeat"], how="left")
        speedup_rows["speedup_vs_baseline"] = (
            speedup_rows["baseline_elapsed_seconds"] / speedup_rows["elapsed_seconds"]
        )

        speedup_summary = speedup_rows.groupby(["tag", "n"], as_index=False).agg(
            speedup_median=("speedup_vs_baseline", "median"),
            speedup_mean=("speedup_vs_baseline", "mean"),
            speedup_std=("speedup_vs_baseline", "std"),
        )

        return summary.merge(speedup_summary, on=["tag", "n"], how="left")

    def _save_plot(self, filename):
        """Apply common formatting and save the current matplotlib figure."""
        plt.tight_layout()
        plt.savefig(self.output_dir / filename, dpi=200)
        plt.close()

    def runtime_vs_n_by_method(self):
        """Plot median runtime against matrix size for each implementation."""
        plt.figure(figsize=(8, 5.5))

        for tag, group in self.summary.groupby("tag"):
            ordered = group.sort_values("n")
            plt.errorbar(
                ordered["n"],
                ordered["elapsed_median"],
                yerr=ordered["elapsed_std"].fillna(0.0),
                marker="s",
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

    def speedup_vs_n(self):
        """Plot median speedup relative to the baseline implementation."""
        if not self.has_baseline:
            print("Skipping speedup plot because baseline data is not present.")
            return

        plt.figure(figsize=(8, 5.5))

        for tag, group in self.summary.groupby("tag"):
            ordered = group.sort_values("n")
            plt.errorbar(
                ordered["n"],
                ordered["speedup_median"],
                yerr=ordered["speedup_std"].fillna(0.0),
                marker="s",
                capsize=3,
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

    def cubic_scaling_check_by_method(self):
        """Plot normalised runtime growth against a cubic reference on log-log axes."""
        plt.figure(figsize=(8, 5.5))

        for tag, group in self.summary.groupby("tag"):
            ordered = group.sort_values("n")

            reference_n = ordered["n"].iloc[0]
            reference_time = ordered["elapsed_median"].iloc[0]
            normalised_runtime = ordered["elapsed_median"] / reference_time
            normalised_std = (
                normalised_runtime
                * ordered["elapsed_std"].fillna(0.0)
                / ordered["elapsed_median"]
            )

            plt.errorbar(
                ordered["n"] / reference_n,
                normalised_runtime,
                yerr=normalised_std.fillna(0.0),
                marker="s",
                capsize=3,
                label=tag,
            )

        all_n = sorted(self.summary["n"].unique())
        if all_n:
            reference_n = float(all_n[0])
            x_values = [float(n) / reference_n for n in all_n]
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
        self._save_plot("cubic_scaling_check_by_method.png")

    def plot_all(self):
        """Run the full comparison plotting pipeline."""
        legacy_plot = self.output_dir / "big_o_check_by_method.png"
        if legacy_plot.exists():
            legacy_plot.unlink()

        speedup_plot = self.output_dir / "speedup_vs_baseline.png"
        if speedup_plot.exists() and not self.has_baseline:
            speedup_plot.unlink()

        self.runtime_vs_n_by_method()
        self.speedup_vs_n()
        self.cubic_scaling_check_by_method()


def main():
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
