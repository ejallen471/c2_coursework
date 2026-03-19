"""Plot benchmark CSV outputs with a shared style and CLI."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

STYLE_FILE = Path(__file__).with_name("pythonStyle.mplstyle")
plt.style.use(STYLE_FILE)

SUMMARY_COLUMNS = {"method", "elapsed_median", "elapsed_mean", "elapsed_error"}

class CholeskyPlotter:
    """
    Create the benchmark plots used by the coursework workflow.

    The class reads the benchmark summary CSV contracts produced by the benchmark modes.

    Notes:
        Passing a raw CSV path is allowed, but the matching `_summary.csv` file must exist.
    """

    SUMMARY_COLUMNS = SUMMARY_COLUMNS


    def __init__(self, style_file: Path | None = None):
        """
        Initialise the plotter with the project Matplotlib style.

        Args:
            style_file: Optional path to a `.mplstyle` file.

        Notes:
            The style file must include the custom `# cholesky.*` metadata entries.
        """
        self.style_file = Path(style_file) if style_file is not None else STYLE_FILE
        self.style_values = self._load_style_values()

    def _load_style_values(self) -> dict[str, float]:
        """
        Load the plotting constants used by the shared plotting routines.

        Returns:
            Mapping of line, marker, and error-bar settings.

        Notes:
            The values come partly from Matplotlib rcParams and partly from custom metadata.
        """
        metadata: dict[str, float] = {}

        for line in self.style_file.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped.startswith("# cholesky."):
                continue

            key_text, value_text = stripped[len("# cholesky."):].split(":", 1)
            metadata[key_text.strip()] = float(value_text.strip())

        required_metadata = {
            "errorbar.linewidth",
            "errorbar.capthick",
        }
        missing_metadata = required_metadata.difference(metadata)
        if missing_metadata:
            missing_text = ", ".join(sorted(missing_metadata))
            raise ValueError(f"Missing required style metadata: {missing_text}")

        return {
            "line_width": float(plt.rcParams["lines.linewidth"]),
            "marker_size": float(plt.rcParams["lines.markersize"]),
            "marker_edge_width": float(plt.rcParams["lines.markeredgewidth"]),
            "error_cap_size": float(plt.rcParams["errorbar.capsize"]),
            "error_line_width": metadata["errorbar.linewidth"],
            "error_cap_thickness": metadata["errorbar.capthick"],
        }

    def _validate_columns(self, df: pd.DataFrame, required_columns: set[str]) -> None:
        """
        Check that a CSV-derived dataframe contains the expected columns.

        Args:
            df: Parsed CSV content.
            required_columns: Columns that must be present.

        Notes:
            A `ValueError` is raised when any required column is missing.
        """
        missing_columns = required_columns.difference(df.columns)
        if missing_columns:
            missing_str = ", ".join(sorted(missing_columns))
            raise ValueError(f"Missing required CSV columns: {missing_str}")

    def _load_plot_data(
        self,
        input_csv: str | Path,
        summary_columns: set[str],
    ) -> pd.DataFrame:
        """
        Load plot input data from a benchmark summary CSV contract.

        Args:
            input_csv: Raw benchmark CSV path or direct summary CSV path.
            summary_columns: Columns expected in the summary form.

        Returns:
            Summary data ready for plotting.

        Notes:
            The plotter consumes the benchmark summary contract rather than recomputing aggregates.
        """
        input_path = Path(input_csv)
        summary_csv = (
            input_path
            if input_path.name.endswith("_summary.csv")
            else input_path.with_name(f"{input_path.stem}_summary.csv")
        )

        if not summary_csv.exists():
            raise FileNotFoundError(
                f"Missing summary CSV for plot input: {summary_csv}"
            )

        df = pd.read_csv(summary_csv)
        self._validate_columns(df, summary_columns)
        return df

    def _plot_grouped_runtime(
        self,
        df: pd.DataFrame,
        group_column: str,
        x_column: str,
        xlabel: str,
        title: str,
        output_path: Path,
        figure_size: tuple[float, float],
        *,
        xscale: str | None = None,
        yscale: str | None = None,
        disable_minor_ticks: bool = False,
    ) -> None:
        """
        Plot grouped median runtimes with error bars and save the figure.

        Args:
            df: Preprocessed plot data.
            group_column: Column used to split lines.
            x_column: Column placed on the x-axis.
            xlabel: Label for the x-axis.
            title: Plot title.
            output_path: Destination image path.
            figure_size: Figure size in inches.
            xscale: Optional x-axis scale.
            yscale: Optional y-axis scale.
            disable_minor_ticks: Whether to disable minor ticks.

        Notes:
            The plot is written to disk and the Matplotlib figure is closed afterwards.
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.figure(figsize=figure_size)

        for group_name, group in df.groupby(group_column):
            ordered = group.sort_values(x_column)
            line = plt.plot(
                ordered[x_column],
                ordered["elapsed_median"],
                marker="x",
                markersize=self.style_values["marker_size"],
                markeredgewidth=self.style_values["marker_edge_width"],
                linewidth=self.style_values["line_width"],
                zorder=3,
                label=group_name,
            )[0]
            plt.errorbar(
                ordered[x_column],
                ordered["elapsed_median"],
                yerr=ordered["elapsed_error"].fillna(0.0),
                fmt="none",
                color=line.get_color(),
                elinewidth=self.style_values["error_line_width"],
                capsize=self.style_values["error_cap_size"],
                capthick=self.style_values["error_cap_thickness"],
                zorder=2,
            )

        if xscale is not None:
            plt.xscale(xscale)
        if yscale is not None:
            plt.yscale(yscale)

        plt.xlabel(xlabel)
        plt.ylabel("Median runtime (s)")
        plt.title(title)
        plt.grid(True, which="major", axis="both")
        if disable_minor_ticks:
            plt.minorticks_off()
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_path, dpi=200)
        plt.close()
        print(f'Successful and saved in "{output_path}"')

    def plot_matrix_size(self, input_csv: str | Path, output_dir: str | Path) -> None:
        """
        Plot runtime against matrix size for one method.

        Args:
            input_csv: Raw or summary CSV for one matrix-size sweep.
            output_dir: Directory for the generated figure.

        Notes:
            The output file is named `runtime_vs_n.png`.
        """
        df = self._load_plot_data(
            input_csv,
            summary_columns=self.SUMMARY_COLUMNS | {"n"},
        )
        self._plot_grouped_runtime(
            df=df,
            group_column="method",
            x_column="n",
            xlabel="Matrix size n",
            title="Runtime vs matrix size",
            output_path=Path(output_dir) / "runtime_vs_n.png",
            figure_size=(7, 5),
            xscale="log",
            yscale="log",
            disable_minor_ticks=True,
        )

    def plot_matrix_size_comparison(self, input_csvs: list[str | Path], output_dir: str | Path) -> None:
        """
        Plot runtime against matrix size for several methods.

        Args:
            input_csvs: Raw or summary CSV files, usually one per method.
            output_dir: Directory for the generated figure.

        Notes:
            The output file is named `runtime_vs_n_by_method.png`.
        """
        df = pd.concat(
            [
                self._load_plot_data(
                    input_csv,
                    summary_columns=self.SUMMARY_COLUMNS | {"n"},
                )
                for input_csv in input_csvs
            ],
            ignore_index=True,
        )
        self._plot_grouped_runtime(
            df=df,
            group_column="method",
            x_column="n",
            xlabel="Matrix size n",
            title="Runtime vs matrix size",
            output_path=Path(output_dir) / "runtime_vs_n_by_method.png",
            figure_size=(8, 5.5),
            xscale="log",
            yscale="log",
            disable_minor_ticks=True,
        )

    def plot_block_size(self, input_csv: str | Path, output_dir: str | Path) -> None:
        """
        Plot runtime against block size for blocked methods.

        Args:
            input_csv: Raw or summary CSV for a block-size sweep.
            output_dir: Directory for the generated figure.

        Notes:
            The output file is named `runtime_vs_block_size.png`.
        """
        df = self._load_plot_data(
            input_csv,
            summary_columns=self.SUMMARY_COLUMNS | {"block_size"},
        )
        self._plot_grouped_runtime(
            df=df,
            group_column="method",
            x_column="block_size",
            xlabel="Block size",
            title="Runtime vs block size",
            output_path=Path(output_dir) / "runtime_vs_block_size.png",
            figure_size=(7, 5),
        )

    def plot_thread_count(self, input_csv: str | Path, output_dir: str | Path) -> None:
        """
        Plot runtime against OpenMP thread count.

        Args:
            input_csv: Raw or summary CSV for a thread-count sweep.
            output_dir: Directory for the generated figure.

        Notes:
            The output file is named `runtime_vs_thread_count.png`.
        """
        df = self._load_plot_data(
            input_csv,
            summary_columns=self.SUMMARY_COLUMNS | {"threads"},
        )
        self._plot_grouped_runtime(
            df=df,
            group_column="method",
            x_column="threads",
            xlabel="Thread count",
            title="Runtime vs thread count",
            output_path=Path(output_dir) / "runtime_vs_thread_count.png",
            figure_size=(7, 5),
        )

    @staticmethod
    def usage() -> str:
        """
        Return the command-line usage text for the plotting CLI.

        Returns:
            Multi-line usage text for all supported plotting commands.
        """
        return (
            "Usage:\n"
            "  python plot/cholesky_plotter.py matrix-size <input_csv> <output_dir>\n"
            "  python plot/cholesky_plotter.py matrix-size-comparison <output_dir> <csv1> [csv2 ...]\n"
            "  python plot/cholesky_plotter.py block-size <input_csv> <output_dir>\n"
            "  python plot/cholesky_plotter.py thread-count <input_csv> <output_dir>"
        )

    @classmethod
    def run_cli(cls, argv: list[str]) -> int:
        """
        Dispatch the plotting CLI to the requested plotting command.

        Args:
            argv: Command-line arguments after the script name.

        Returns:
            Process-style exit code on success.

        Notes:
            Invalid argument shapes raise `SystemExit` with the shared usage text.
        """
        if not argv:
            raise SystemExit(cls.usage())

        command = argv[0]
        plotter = cls()

        if command == "matrix-size":
            if len(argv) != 3:
                raise SystemExit(cls.usage())

            plotter.plot_matrix_size(argv[1], argv[2])
            return 0

        if command == "matrix-size-comparison":
            if len(argv) < 3:
                raise SystemExit(cls.usage())

            plotter.plot_matrix_size_comparison(argv[2:], argv[1])
            return 0

        if command == "block-size":
            if len(argv) != 3:
                raise SystemExit(cls.usage())

            plotter.plot_block_size(argv[1], argv[2])
            return 0

        if command == "thread-count":
            if len(argv) != 3:
                raise SystemExit(cls.usage())

            plotter.plot_thread_count(argv[1], argv[2])
            return 0

        raise SystemExit(cls.usage())


if __name__ == "__main__":
    raise SystemExit(CholeskyPlotter.run_cli(sys.argv[1:]))
