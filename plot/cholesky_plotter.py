"""Plot benchmark CSV outputs with a shared style and CLI."""

import sys
from pathlib import Path

import matplotlib.colors as mcolors
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

    def _load_style_values(self) -> dict[str, float | str]:
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
            "highlight.markersize",
            "highlight.markeredgewidth",
        }
        missing_metadata = required_metadata.difference(metadata)
        if missing_metadata:
            missing_text = ", ".join(sorted(missing_metadata))
            raise ValueError(f"Missing required style metadata: {missing_text}")

        return {
            "line_width": float(plt.rcParams["lines.linewidth"]),
            "marker": str(plt.rcParams["lines.marker"]),
            "marker_size": float(plt.rcParams["lines.markersize"]),
            "marker_edge_width": float(plt.rcParams["lines.markeredgewidth"]),
            "error_cap_size": float(plt.rcParams["errorbar.capsize"]),
            "error_line_width": metadata["errorbar.linewidth"],
            "error_cap_thickness": metadata["errorbar.capthick"],
            "highlight_marker_size": metadata["highlight.markersize"],
            "highlight_marker_edge_width": metadata["highlight.markeredgewidth"],
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

    @staticmethod
    def _title_with_matrix_size(df: pd.DataFrame, base_title: str) -> str:
        """
        Append the matrix size to a plot title when the data uses one size.

        Args:
            df: Plot data that may contain an `n` column.
            base_title: Title text before any matrix-size suffix.

        Returns:
            Title text with an `n = ...` suffix when appropriate.
        """
        if "n" not in df.columns or df["n"].nunique() != 1:
            return base_title

        n_value = df["n"].iloc[0]
        if float(n_value).is_integer():
            n_text = str(int(n_value))
        else:
            n_text = str(n_value)

        return f"{base_title} (n = {n_text})"

    @staticmethod
    def _darken_color(color: str, factor: float = 0.7) -> tuple[float, float, float]:
        """
        Return a darker shade of a Matplotlib colour.

        Args:
            color: Matplotlib-compatible colour value.
            factor: Multiplicative darkening factor in `[0, 1]`.

        Returns:
            Darkened RGB tuple.
        """
        red, green, blue = mcolors.to_rgb(color)
        return (red * factor, green * factor, blue * factor)

    @staticmethod
    def _matrix_size_suffix(df: pd.DataFrame) -> str:
        """
        Return a `_n...` suffix when the data uses one matrix size.

        Args:
            df: Plot data that may contain an `n` column.

        Returns:
            Suffix text for directory naming, or an empty string.
        """
        if "n" not in df.columns or df["n"].nunique() != 1:
            return ""

        n_value = df["n"].iloc[0]
        if float(n_value).is_integer():
            return f"_n{int(n_value)}"

        return f"_n{n_value}"

    def _resolve_output_dir(
        self,
        input_csv: str | Path,
        output_dir: str | Path | None,
        df: pd.DataFrame,
    ) -> Path:
        """
        Resolve the destination directory for generated figures.

        Args:
            input_csv: Input CSV path used for the plot.
            output_dir: Optional user-supplied output directory.
            df: Plot data used to infer matrix size when needed.

        Returns:
            Directory where the figure should be written.

        Notes:
            When `output_dir` is omitted, plots under a `Raw/` tree are written to the
            matching `Figures/` tree and include the matrix size automatically when needed.
        """
        if output_dir is not None:
            return Path(output_dir)

        input_path = Path(input_csv)
        if "Raw" in input_path.parts:
            raw_index = input_path.parts.index("Raw")
            figures_root = Path(*input_path.parts[:raw_index], "Figures")
            directory_name = input_path.parent.name
            matrix_size_suffix = self._matrix_size_suffix(df)
            if matrix_size_suffix and not directory_name.endswith(matrix_size_suffix):
                directory_name = f"{directory_name}{matrix_size_suffix}"
            return figures_root / directory_name

        return input_path.parent / "figures"

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
        highlight_minimum: bool = False,
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
            highlight_minimum: Whether to emphasise the minimum-runtime point per group.

        Notes:
            The plot is written to disk and the Matplotlib figure is closed afterwards.
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure, axis = plt.subplots(figsize=figure_size)

        self._draw_grouped_runtime(
            axis=axis,
            df=df,
            group_column=group_column,
            x_column=x_column,
            highlight_minimum=highlight_minimum,
            include_labels=True,
        )

        if xscale is not None:
            axis.set_xscale(xscale)
        if yscale is not None:
            axis.set_yscale(yscale)

        axis.set_xlabel(xlabel)
        axis.set_ylabel("Median runtime (s)")
        axis.set_title(title)
        axis.grid(True, which="major", axis="both")
        if disable_minor_ticks:
            axis.minorticks_off()
        axis.legend()
        figure.tight_layout()
        figure.savefig(output_path, dpi=200)
        plt.close(figure)
        print(f'Successful and saved in "{output_path}"')

    def _draw_grouped_runtime(
        self,
        axis,
        df: pd.DataFrame,
        group_column: str,
        x_column: str,
        *,
        highlight_minimum: bool,
        include_labels: bool,
        color_by_group: dict[str, str] | None = None,
        marker_size: float | None = None,
        marker_edge_width: float | None = None,
        error_line_width: float | None = None,
        error_cap_thickness: float | None = None,
        highlight_marker_size: float | None = None,
        highlight_marker_edge_width: float | None = None,
        highlight_color_mode: str = "black",
    ) -> None:
        """
        Draw grouped runtime curves on an existing Matplotlib axis.

        Args:
            axis: Destination Matplotlib axis.
            df: Preprocessed plot data.
            group_column: Column used to split lines.
            x_column: Column placed on the x-axis.
            highlight_minimum: Whether to emphasise the minimum-runtime point per group.
            include_labels: Whether to include legend labels on the drawn series.
            color_by_group: Optional fixed colour mapping shared across axes.
            marker_size: Optional marker-size override for this plot only.
            marker_edge_width: Optional marker-edge width override for this plot only.
            error_line_width: Optional error-bar line width override for this plot only.
            error_cap_thickness: Optional error-bar cap thickness override for this plot only.
            highlight_marker_size: Optional highlight marker-size override for this plot only.
            highlight_marker_edge_width: Optional highlight marker-edge width override.
            highlight_color_mode: Whether the highlight marker is black or matches the series.
        """
        resolved_marker_size = (
            self.style_values["marker_size"] if marker_size is None else marker_size
        )
        resolved_marker_edge_width = (
            self.style_values["marker_edge_width"]
            if marker_edge_width is None
            else marker_edge_width
        )
        resolved_error_line_width = (
            self.style_values["error_line_width"]
            if error_line_width is None
            else error_line_width
        )
        resolved_error_cap_thickness = (
            self.style_values["error_cap_thickness"]
            if error_cap_thickness is None
            else error_cap_thickness
        )
        resolved_highlight_marker_size = (
            self.style_values["highlight_marker_size"]
            if highlight_marker_size is None
            else highlight_marker_size
        )
        resolved_highlight_marker_edge_width = (
            self.style_values["highlight_marker_edge_width"]
            if highlight_marker_edge_width is None
            else highlight_marker_edge_width
        )

        for group_name, group in df.groupby(group_column):
            ordered = group.sort_values(x_column)
            label = group_name if include_labels else "_nolegend_"
            line_color = None if color_by_group is None else color_by_group.get(group_name)
            errorbar_container = axis.errorbar(
                ordered[x_column],
                ordered["elapsed_median"],
                yerr=ordered["elapsed_error"].fillna(0.0),
                fmt="-",
                marker=self.style_values["marker"],
                markersize=resolved_marker_size,
                markeredgewidth=resolved_marker_edge_width,
                linewidth=self.style_values["line_width"],
                elinewidth=resolved_error_line_width,
                capsize=self.style_values["error_cap_size"],
                capthick=resolved_error_cap_thickness,
                zorder=3,
                label=label,
                color=line_color,
            )

            if highlight_minimum:
                minimum_row = ordered.loc[ordered["elapsed_median"].idxmin()]
                highlight_color = (
                    self._darken_color(errorbar_container.lines[0].get_color())
                    if highlight_color_mode == "series"
                    else "black"
                )
                axis.plot(
                    minimum_row[x_column],
                    minimum_row["elapsed_median"],
                    linestyle="none",
                    marker=self.style_values["marker"],
                    markersize=resolved_highlight_marker_size,
                    markeredgewidth=resolved_highlight_marker_edge_width,
                    color=highlight_color,
                    zorder=4,
                )

    def _plot_block_size_with_inset(
        self,
        df: pd.DataFrame,
        title: str,
        output_path: Path,
        figure_size: tuple[float, float],
    ) -> None:
        """
        Plot the OpenMP block-size comparison with an inset that omits the DAG method.

        Args:
            df: Preprocessed block-size summary data.
            title: Plot title.
            output_path: Destination image path.
            figure_size: Figure size in inches.
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure, axis = plt.subplots(figsize=figure_size)
        colour_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
        ordered_methods = list(df["method"].drop_duplicates())
        color_by_group = {
            method_name: colour_cycle[index % len(colour_cycle)]
            for index, method_name in enumerate(ordered_methods)
        } if colour_cycle else None

        self._draw_grouped_runtime(
            axis=axis,
            df=df,
            group_column="method",
            x_column="block_size",
            highlight_minimum=True,
            include_labels=True,
            color_by_group=color_by_group,
            marker_size=2.3,
            marker_edge_width=0.75,
            error_line_width=0.18,
            error_cap_thickness=0.18,
            highlight_marker_size=5.2,
            highlight_marker_edge_width=0.95,
            highlight_color_mode="series",
        )

        axis.set_xlabel("Block size")
        axis.set_ylabel("Median runtime (s)")
        axis.set_title(title)
        axis.grid(True, which="major", axis="both")
        axis.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.24),
            ncol=2,
        )

        inset_df = df[df["method"] != "openmp_task_dag_blocked"]
        if not inset_df.empty:
            inset_axis = axis.inset_axes([60.0, 40.0, 190.0, 90.0], transform=axis.transData)
            inset_axis.set_facecolor("white")
            self._draw_grouped_runtime(
                axis=inset_axis,
                df=inset_df,
                group_column="method",
                x_column="block_size",
                highlight_minimum=True,
                include_labels=False,
                color_by_group=color_by_group,
                marker_size=2.3,
                marker_edge_width=0.75,
                error_line_width=0.18,
                error_cap_thickness=0.18,
                highlight_marker_size=5.2,
                highlight_marker_edge_width=0.95,
                highlight_color_mode="series",
            )
            inset_axis.grid(False)
            inset_axis.tick_params(labelsize=10)

        figure.tight_layout(rect=(0, 0.14, 1, 1))
        figure.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close(figure)
        print(f'Successful and saved in "{output_path}"')

    def plot_matrix_size(
        self, input_csv: str | Path, output_dir: str | Path | None = None
    ) -> None:
        """
        Plot runtime against matrix size for one method.

        Args:
            input_csv: Raw or summary CSV for one matrix-size sweep.
            output_dir: Optional directory for the generated figure.

        Notes:
            The output file is named `runtime_vs_n.png`.
        """
        df = self._load_plot_data(
            input_csv,
            summary_columns=self.SUMMARY_COLUMNS | {"n"},
        )
        resolved_output_dir = self._resolve_output_dir(input_csv, output_dir, df)
        self._plot_grouped_runtime(
            df=df,
            group_column="method",
            x_column="n",
            xlabel="Matrix size n",
            title="Runtime vs matrix size",
            output_path=resolved_output_dir / "runtime_vs_n.png",
            figure_size=(7, 5),
            xscale="log",
            yscale="log",
            disable_minor_ticks=True,
        )

    def plot_matrix_size_comparison(
        self, input_csvs: list[str | Path], output_dir: str | Path | None = None
    ) -> None:
        """
        Plot runtime against matrix size for several methods.

        Args:
            input_csvs: Raw or summary CSV files, usually one per method.
            output_dir: Optional directory for the generated figure.

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
        resolved_output_dir = self._resolve_output_dir(input_csvs[0], output_dir, df)
        self._plot_grouped_runtime(
            df=df,
            group_column="method",
            x_column="n",
            xlabel="Matrix size n",
            title="Runtime vs matrix size",
            output_path=resolved_output_dir / "runtime_vs_n_by_method.png",
            figure_size=(8, 5.5),
            yscale="log",
        )

    def plot_block_size(
        self, input_csv: str | Path, output_dir: str | Path | None = None
    ) -> None:
        """
        Plot runtime against block size for blocked methods.

        Args:
            input_csv: Raw or summary CSV for a block-size sweep.
            output_dir: Optional directory for the generated figure.

        Notes:
            The output file is named `runtime_vs_block_size.png`.
        """
        df = self._load_plot_data(
            input_csv,
            summary_columns=self.SUMMARY_COLUMNS | {"block_size"},
        )
        resolved_output_dir = self._resolve_output_dir(input_csv, output_dir, df)
        title = self._title_with_matrix_size(df, "Runtime vs block size")

        if "openmp_task_dag_blocked" in set(df["method"]):
            self._plot_block_size_with_inset(
                df=df,
                title=title,
                output_path=resolved_output_dir / "runtime_vs_block_size.png",
                figure_size=(7, 5),
            )
            return

        self._plot_grouped_runtime(
            df=df,
            group_column="method",
            x_column="block_size",
            xlabel="Block size",
            title=title,
            output_path=resolved_output_dir / "runtime_vs_block_size.png",
            figure_size=(7, 5),
            highlight_minimum=True,
        )

    def plot_thread_count(
        self, input_csv: str | Path, output_dir: str | Path | None = None
    ) -> None:
        """
        Plot runtime against OpenMP thread count.

        Args:
            input_csv: Raw or summary CSV for a thread-count sweep.
            output_dir: Optional directory for the generated figure.

        Notes:
            The output file is named `runtime_vs_thread_count.png`.
        """
        df = self._load_plot_data(
            input_csv,
            summary_columns=self.SUMMARY_COLUMNS | {"threads"},
        )
        resolved_output_dir = self._resolve_output_dir(input_csv, output_dir, df)
        self._plot_grouped_runtime(
            df=df,
            group_column="method",
            x_column="threads",
            xlabel="Thread count",
            title=self._title_with_matrix_size(df, "Runtime vs thread count"),
            output_path=resolved_output_dir / "runtime_vs_thread_count.png",
            figure_size=(7, 5),
            highlight_minimum=True,
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
            "  python plot/cholesky_plotter.py matrix-size <input_csv> [output_dir]\n"
            "  python plot/cholesky_plotter.py matrix-size-comparison <output_dir> <csv1> [csv2 ...]\n"
            "  python plot/cholesky_plotter.py matrix-size-comparison <csv1> [csv2 ...]\n"
            "  python plot/cholesky_plotter.py block-size <input_csv> [output_dir]\n"
            "  python plot/cholesky_plotter.py thread-count <input_csv> [output_dir]"
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
            if len(argv) not in {2, 3}:
                raise SystemExit(cls.usage())

            plotter.plot_matrix_size(argv[1], argv[2] if len(argv) == 3 else None)
            return 0

        if command == "matrix-size-comparison":
            if len(argv) < 2:
                raise SystemExit(cls.usage())

            if argv[1].endswith(".csv"):
                plotter.plot_matrix_size_comparison(argv[1:], None)
                return 0

            if len(argv) < 3:
                raise SystemExit(cls.usage())

            plotter.plot_matrix_size_comparison(argv[2:], argv[1])
            return 0

        if command == "block-size":
            if len(argv) not in {2, 3}:
                raise SystemExit(cls.usage())

            plotter.plot_block_size(argv[1], argv[2] if len(argv) == 3 else None)
            return 0

        if command == "thread-count":
            if len(argv) not in {2, 3}:
                raise SystemExit(cls.usage())

            plotter.plot_thread_count(argv[1], argv[2] if len(argv) == 3 else None)
            return 0

        raise SystemExit(cls.usage())


if __name__ == "__main__":
    raise SystemExit(CholeskyPlotter.run_cli(sys.argv[1:]))
