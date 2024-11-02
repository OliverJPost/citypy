import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import List, Literal, Optional, Type

import geopandas as gpd
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import typer
from fitter import Fitter
from loguru import logger
from matplotlib.patches import ConnectionPatch, Rectangle
from matplotlib.ticker import PercentFormatter
from scipy.stats import cauchy, lognorm
from sklearn.feature_selection import mutual_info_classif

from citypy.cli import app
from citypy.cli.commands.gridify import grid_to_numpy, gridify
from citypy.contextualize.road_neighborhood import find_edge_neighborhood
from citypy.util.graph import graph_from_edges_only


class InfluencedHistogramPlot:
    histogram: ...
    influences: list[str]
    title: ...
    bins: list["Bin"]

    def __init__(self, bins: list["Bin"], title, influences):
        self.bins = bins
        self.title = title
        self.influences = influences

    def plot(self):
        num_subplot_rows = len(self.influences)

        fig = plt.figure(figsize=(12, 12))
        gs = fig.add_gridspec(
            3 + num_subplot_rows,
            2,
            height_ratios=[3, 0.5, 0.5] + [0.8] * num_subplot_rows,
            width_ratios=[0.8, 0.2],
            hspace=0,
        )

        fig.suptitle(self.title, fontsize=16, fontweight="bold")

        ax_hist = self.plot_histogram(fig, gs)
        self.plot_whitespace(fig, gs)

        self.plot_influence_rows(ax_hist, fig, gs)

        # Adjust the layout
        fig.tight_layout()
        plt.subplots_adjust(left=0.2, top=0.93, bottom=0.05, hspace=0.2)
        plt.show()

    def plot_influence_rows(self, ax_hist, fig, gs):
        # Add header for subplot rows
        ax_header = fig.add_subplot(gs[2])
        ax_header.text(
            -0.05,
            0.5,
            "Influences",
            fontsize=14,
            fontweight="bold",
            va="center",
            ha="right",
        )
        ax_header.axis("off")
        # Calculate the width of each bin
        bin_width = (
            self.bins[0].bounds[1] - self.bins[0].bounds[0]
        )  # TODO dynamic bin widths
        # Create subplots for each row

        all_bin_data = pd.concat([b.bin_data for b in self.bins])
        # bin_id_data = []
        # for i, b in enumerate(self.bins):
        #     bin_data = b.bin_data.dropna()
        #     bin_data["bin_id"] = i
        #     bin_id_data.append(bin_data)
        # y = pd.concat(bin_id_data)
        # X = np.empty((len(y), len(self.influences)))

        for row, influence_name in enumerate(self.influences):
            ax_small = fig.add_subplot(gs[row + 3, 0])
            ax_small.set_xlim(self.bins[0].bounds[0], self.bins[-1].bounds[1])
            ax_small.set_ylim(0, 1)
            ax_small.axis("off")

            # Add label for the row
            ax_small.text(
                -0.05,
                0.5,
                influence_name,
                va="center",
                ha="right",
                transform=ax_small.transAxes,
            )
            influence_distributions = [
                getattr(b.influences, influence_name) for b in self.bins
            ]
            all_influence_data = pd.concat(
                [distr.data for distr in influence_distributions]
            )
            # X[:, row] = all_influence_data[y.index].to_numpy()
            min_95th = all_influence_data.quantile(0.05)
            max_95th = all_influence_data.quantile(0.95)
            y_limits = (min_95th if min_95th < 0 else 0, max_95th)

            # New subplot to the right of ax_small
            ax_right = fig.add_subplot(gs[row + 3, 1])  # New axis in the next column
            ax_right.margins(y=0.1)
            ax_right.scatter(all_bin_data, all_influence_data, alpha=0.05)
            ax_right.set_ylim(*y_limits)
            ax_right.invert_yaxis()

            for i, this_bin in enumerate(self.bins):
                # Calculate the position for each small plot
                left = this_bin.bounds[0]
                bottom = 0.1
                width = bin_width
                height = 0.8

                # Create a small axes for each bin
                ax_bin = ax_small.inset_axes(
                    [left, bottom, width, height], transform=ax_small.transData
                )

                getattr(this_bin.influences, influence_name).plot(ax_bin, y_limits)

                # ax_bin.set_xticks([])
                if i != 0:
                    ax_bin.set_yticks([])

                # Only last row
                if row == len(self.influences) - 1:
                    # Connection patch from bottom left to top right corner of subplot
                    con = ConnectionPatch(
                        xyA=(this_bin.bounds[0], 0),
                        coordsA=ax_hist.transData,
                        xyB=(0, 0),
                        coordsB=ax_bin.transAxes,
                        arrowstyle="-",
                        linestyle="dotted",
                        color="gray",
                        linewidth=1,
                    )
                    fig.add_artist(con)

                # Only last column
                if i == len(self.bins) - 2:
                    # Connection patch from top right to top left corner of next subplot
                    con = ConnectionPatch(
                        xyA=(this_bin.bounds[1], 0),
                        coordsA=ax_hist.transData,
                        xyB=(1, 0),
                        coordsB=ax_bin.transAxes,
                        arrowstyle="-",
                        linestyle="dotted",
                        color="gray",
                        linewidth=1,
                    )
                    fig.add_artist(con)

            # mi = mutual_info_classif(X, y)
            # print(mi)

    def plot_whitespace(self, fig, gs):
        fig.add_subplot(gs[1, 0]).set_visible(False)
        ax_header = fig.add_subplot(gs[2])
        ax_header.text(
            -0.05,
            0.5,
            "Influences",
            fontsize=14,
            fontweight="bold",
            va="center",
            ha="right",
        )
        ax_header.axis("off")

    def plot_histogram(self, fig, gs):
        # Create the main histogram
        ax_hist = fig.add_subplot(gs[0, 0])
        ax_hist.margins(x=0)
        bin_borders = [b.bounds[0] for b in self.bins] + [self.bins[-1].bounds[1]]
        print(bin_borders)
        bin_data = [b.bin_data for b in self.bins]
        bin_data = pd.concat(bin_data)

        # Calculate histogram data with density=True
        counts, bins = np.histogram(bin_data, bins=bin_borders, density=True)
        bin_widths = np.diff(bins)

        # Convert density to probability by multiplying by bin width
        probabilities = counts * bin_widths

        # Clear the existing histogram and plot the new histogram with probabilities
        ax_hist.cla()
        ax_hist.bar(
            bins[:-1], probabilities, width=bin_widths, align="edge", alpha=0.75
        )

        # Add x and y labels to the histogram
        ax_hist.set_xlabel("Value", fontsize=12)
        ax_hist.set_ylabel("Probability", fontsize=12)

        # Set x-axis limits to the range of bin borders
        ax_hist.set_xlim([bin_borders[0], bin_borders[-1]])

        # Set y-axis to display probabilities
        ax_hist.yaxis.set_major_formatter(PercentFormatter(1))

        # # Set y-axis ticks (logical places)
        # ax_hist.set_yticks(
        #     [i / 10 for i in range(0, 10)]
        # )  # Sets ticks at 0%, 10%, 20%, ..., 50%

        # Show only bottom and left spines
        ax_hist.spines["top"].set_visible(False)
        ax_hist.spines["right"].set_visible(False)

        return ax_hist


class FitQuality(Enum):
    GOOD = 0
    OK = 1
    BAD = 2
    UNDEFINED = 3


class Distribution:
    def __init__(self, data: pd.Series):
        self.data = data

        if len(data.dropna()) < 2:
            self.distribution = ZeroDistribution()
            self.sumsquare_error = None
            self.ks_statistic = None
            self.ks_p_value = None
            self.quality = FitQuality.UNDEFINED
            return

        f = Fitter(data.dropna(), distributions=["cauchy", "lognorm"])
        f.fit()
        best_distribution, params = [x for x in f.get_best().items()][0]
        self.sumsquare_error = f.df_errors["sumsquare_error"][best_distribution]
        self.ks_statistic = f.df_errors["ks_statistic"][best_distribution]
        if self.sumsquare_error < 0.2 and self.ks_statistic < 0.2:
            self.quality = FitQuality.GOOD
        elif self.sumsquare_error < 1.0 and self.ks_statistic < 0.4:
            self.quality = FitQuality.OK
        else:
            self.quality = FitQuality.BAD

        self.median = data.median()
        self.mean = data.mean()
        if best_distribution == "cauchy":
            print(params)
            self.distribution = CauchyDistribution(**params)
        elif best_distribution == "lognorm":
            self.distribution = LogNormDistribution(**params)
        elif best_distribution == "uniform":
            self.distribution = UniformDistribution()
        else:
            raise Exception("Invalid distribution: " + best_distribution)

    def plot(self, ax, ylimits):
        ax.hist(self.data.dropna(), density=True, orientation="horizontal", bins=10)
        self.distribution.plot(ax, ylimits)
        if self.sumsquare_error:
            ax.text(
                1.0,
                1.0,
                f"ε: {self.sumsquare_error:.2f}",
                verticalalignment="top",
                horizontalalignment="right",
                transform=ax.transAxes,
                c="red"
                if self.sumsquare_error > 1.0
                else "orange"
                if self.sumsquare_error > 0.2
                else "green",
            )
            ax.text(
                1.0,
                0.8,
                f"ks: {self.ks_statistic:.2f}",
                verticalalignment="top",
                horizontalalignment="right",
                transform=ax.transAxes,
                c="red"
                if self.ks_statistic > 0.4
                else "orange"
                if self.ks_statistic > 0.2
                else "green",
            )
            ax.text(
                1.0,
                0.6,
                f"Mdn: {self.median:.0f}",
                verticalalignment="top",
                horizontalalignment="right",
                transform=ax.transAxes,
                c="grey",
            )
            ax.text(
                1.0,
                0.4,
                f"M: {self.mean:.0f}",
                verticalalignment="top",
                horizontalalignment="right",
                transform=ax.transAxes,
                c="grey",
            )
        x_factor = -0.001 * (ylimits[1] - ylimits[0]) + 0.45
        ax.set_xlim(0, x_factor)
        ax.invert_yaxis()

    def as_dict(self) -> dict:
        if self.quality == FitQuality.BAD:
            return (
                ZeroDistribution().to_dict()
            )  # FIXME this shouldn't be a zero distribution, but an even distribution
        else:
            return self.distribution.to_dict()


@dataclass
class CauchyDistribution:
    scale: float
    loc: float

    def to_dict(self):
        return {
            "type": "Cauchy",
            "scale": self.scale,
            "loc": self.loc,
        }

    def plot(self, ax, ylimits):
        y = np.linspace(*ylimits, 100)
        x = cauchy.pdf(y, self.loc, self.scale)
        ax.plot(x, y, label="Cauchy", color="blue")
        ax.fill_between(x, y, color="blue", alpha=0.2)


@dataclass
class LogNormDistribution:
    s: float
    loc: float
    scale: float

    def to_dict(self):
        return {
            "type": "LogNormal",
            "s": self.s,
            "loc": self.loc,
            "scale": self.scale,
        }

    def plot(self, ax, ylimits):
        y = np.linspace(*ylimits, 100)
        x = lognorm.pdf(y, self.s, self.loc, self.scale)
        ax.plot(x, y, label="LogNormal", color="red")
        ax.fill_between(x, y, color="red", alpha=0.2)


@dataclass
class UniformDistribution:
    def to_dict(self):
        return {
            "type": "Uniform",
        }

    def plot(self, ax, ylimits):
        ax.hlines(ylimits[0], 0, 0.5, color="green", label="Uniform")


class ZeroDistribution:
    def to_dict(self):
        return {
            "type": "Zero",
        }

    def plot(self, ax, ylimits):
        ax.text(
            0.5,
            0.5,
            "No data",
            fontsize=12,
            ha="center",
            va="center",
            transform=ax.transAxes,
            c="grey",
        )


class ClusterSummaryBuilder:
    samples: list[gpd.GeoSeries]
    sample_type: Literal["street"]
    distributions: list["ProbabilityHistogram"]

    def __init__(self, type: Literal["street"], cluster):
        self.samples = []
        self.cluster = cluster
        self.sample_type = type
        self.distributions = []

    def add_sample(self, sample: gpd.GeoSeries):
        assert sample.geometry.type == "LineString"

        self.samples.append(sample)

    def add_distribution(self, distribution: "ProbabilityHistogram"):
        self.distributions.append(distribution)

    def build_summary(self):
        pass

    def plot(self):
        for distribution in self.distributions:
            influences = [
                "previous_segment_forward_angle",
                "previous_segment_length",
                "distance_to_last_intersection",
            ]
            plot = InfluencedHistogramPlot(
                distribution.bins,
                distribution.name + f"- cluster {self.cluster}",
                influences,
            ).plot()

    def _as_dict(self):
        return {distr.name: distr.as_dict() for distr in self.distributions}

    def export_template(self, fp: Path):
        with open(fp, "w") as f:
            json.dump(self._as_dict(), f)


class ClusterTemplateBuilder:
    def __init__(self):
        pass


class InfluenceMap(ABC):
    @abstractmethod
    def as_dict(self) -> dict[str, Distribution]: ...


class StreetInfluenceMap(InfluenceMap):
    previous_segment_forward_angle: Distribution
    previous_segment_length: Distribution
    distance_to_last_intersection: Distribution

    def __init__(self):
        self.previous_segment_forward_angle = None
        self.previous_segment_length = None
        self.distance_to_last_intersection = None

    def as_dict(self) -> dict[str, Distribution]:
        data = {}
        if self.previous_segment_forward_angle is None:
            raise Exception("No distribution for previous_segment_forward_angle")
        data["previous_segment_forward_angle"] = (
            self.previous_segment_forward_angle.as_dict()
        )

        if self.previous_segment_length is None:
            raise Exception("No distribution for previous_segment_length")
        data["previous_segment_length"] = self.previous_segment_length.as_dict()

        if self.distance_to_last_intersection is None:
            raise Exception("No distribution for distance_to_last_intersection")
        data["distance_to_last_intersection"] = (
            self.distance_to_last_intersection.as_dict()
        )

        return data


class Bin:
    bounds: tuple[float, float]
    median: float
    probability: float
    influences: InfluenceMap
    bin_data: pd.Series

    def __init__(
        self,
        bounds: tuple[float, float],
        bin_data: pd.Series,
        probability: float,
        influence_map_type: Type[InfluenceMap],
    ):
        self.bounds = bounds
        self.median = bin_data.median() if not bin_data.dropna().empty else 0
        self.probability = probability
        self.influences = influence_map_type()
        self.bin_data = bin_data

    def add_influence(self, influence_data: pd.Series, influence_name: str):
        influence_mapped = influence_data[self.bin_data.index]
        distribution = Distribution(influence_mapped)
        setattr(self.influences, influence_name, distribution)

    def as_dict(self):
        return {
            "bounds": self.bounds,
            "median": self.median,
            "probability": self.probability,
            "influences": self.influences.as_dict(),
        }


class ProbabilityHistogram:
    bins: list["Bin"]

    def __init__(
        self, data: pd.Series, name: str, influence_map_type: Type[InfluenceMap]
    ):
        self.name = name
        bins = np.histogram_bin_edges(data.dropna(), bins=10)
        self.bins = []
        counts = []
        for bin_start, bin_end in zip(bins[:-1], bins[1:]):
            bin_data = data[(data >= bin_start) & (data < bin_end)]
            bin = Bin((bin_start, bin_end), bin_data, 0, influence_map_type)
            self.bins.append(bin)
            counts.append(len(bin_data))

        total = sum(counts)
        for bin in self.bins:
            bin.probability = len(bin.bin_data) / total

    def add_influence(self, influence_data: pd.Series, influence_name: str):
        for bin in self.bins:
            bin.add_influence(influence_data, influence_name)

    def plot(self):
        fig, ax = plt.subplots()
        for i, bin in enumerate(self.bins):
            rect = Rectangle(
                (bin.bounds[0], 0),
                bin.bounds[1] - bin.bounds[0],
                bin.probability,
                edgecolor="black",
                facecolor="blue",
                alpha=0.5,
            )
            ax.add_patch(rect)
            ax.text(
                bin.bounds[0] + (bin.bounds[1] - bin.bounds[0]) / 2,
                bin.probability / 2,
                f"{bin.median:.2f}",
                ha="center",
                va="center",
            )

        ax.set_xlabel(self.name)
        ax.set_ylabel("Probability")
        ax.set_title(f"{self.name} Probability Histogram")
        ax.set_xlim(
            min([bin.bounds[0] for bin in self.bins]),
            max([bin.bounds[1] for bin in self.bins]),
        )
        ax.set_ylim(0, 1)
        ax.set_yticks([])
        ax.set_xticks(
            [bin.bounds[0] for bin in self.bins]
            + [(bin.bounds[1] + bin.bounds[0]) / 2 for bin in self.bins]
        )
        ax.grid(axis="x")
        plt.show()

    def as_dict(self):
        return {"bins": [bin.as_dict() for bin in self.bins]}


BUILDING_CLASS_MAPPING = {
    "apartments": 14,
    "big_commercial": 15,
    "complex": 16,
    "detached": 17,
    "filled_block": 18,
    "industrial": 19,
    "irregular_block": 20,
    "perimeter_block": 21,
    "terraced": 22,
    None: 99,
    "NONE": 99,
}


@app.command()
def encode(
    input_gpkg: Path = typer.Argument(
        help="Path to the input GPKG file.",
    ),
    road_cluster_column: str = typer.Option(
        None,
        help="Name of the column containing the road cluster.",
    ),
    building_class_column=typer.Option(
        None,
        help="Name of the column containing the building class.",
    ),
    output_dir: Path = typer.Option(None, "--output-dir", "-o"),
) -> None:
    gpkg_name = input_gpkg.stem

    logger.info("Encoding road clusters")
    gridify(
        input_gpkg,
        layer_name="road_edges",
        most_intersecting_columns=[road_cluster_column],
        mode_columns=[],
        sum_columns=[],
        mean_columns=[],
        count_columns=[],
        overwrite=False,
    )

    logger.info("Encoding building classes")
    gridify(
        input_gpkg,
        layer_name="buildings",
        most_intersecting_columns=[building_class_column],
        mode_columns=[],
        sum_columns=[],
        mean_columns=[],
        count_columns=[],
        overwrite=False,
    )
    logger.info("Encoding city center")
    gridify(
        input_gpkg,
        layer_name="city_center",
        most_intersecting_columns=[],
        mode_columns=[],
        sum_columns=[],
        mean_columns=[],
        count_columns=["name"],
        overwrite=False,
    )

    roads_gdf = gpd.read_file(input_gpkg, layer="road_edges", engine="pyogrio")
    geometry_type = roads_gdf.geom_type[0]
    if geometry_type == "LineString":
        roads_gdf = roads_gdf.set_index(["u", "v", "key"])

    grid = gpd.read_file(input_gpkg, layer="grid")
    grid["road_cluster_int"] = grid[
        f"road_edges::{road_cluster_column}::most_intersecting::mode_kernel3"
    ].apply(lambda value: int(float(value)) if value not in (None, "NONE") else 99)
    grid["building_cluster_int"] = grid[
        f"buildings::{building_class_column}::most_intersecting::mode_kernel3"
    ].map(BUILDING_CLASS_MAPPING)

    street_grid_np = grid_to_numpy(grid, "road_cluster_int").astype(np.int32)
    building_grid_np = grid_to_numpy(grid, "building_cluster_int").astype(np.int32)
    city_center_grid_np = grid_to_numpy(grid, "city_center::name::count").astype(
        np.int32
    )
    # # map unique strings to ids
    # unique_values = values.unique()
    # value_map = {
    #     value: int(float(value))
    #     for idx, value in enumerate(unique_values)
    #     if value is not None
    # }
    # values = values.map(value_map).fillna(-2).to_numpy()
    # values = values.astype(np.int32)
    #
    # bounds = grid.total_bounds
    # width, height = (
    #     grid.geometry.apply(lambda geom: geom.bounds[2] - geom.bounds[0]).mean(),
    #     grid.geometry.apply(lambda geom: geom.bounds[3] - geom.bounds[1]).mean(),
    # )
    # rows = int((bounds[3] - bounds[1]) / height)
    # cols = int((bounds[2] - bounds[0]) / width)
    #
    # assert len(values) == rows * cols

    if not output_dir.exists():
        output_dir.mkdir()
    np.savez(
        output_dir / (gpkg_name + ".npz"),
        cluster_street=street_grid_np,
        building_class=building_grid_np,
        city_center=city_center_grid_np,
    )

    unique_clusters = roads_gdf[road_cluster_column].unique()
    for cluster in unique_clusters:
        cluster_gdf = roads_gdf[roads_gdf[road_cluster_column] == cluster]
        if len(cluster_gdf) == 0:
            continue

        summary_builder = ClusterSummaryBuilder("street", cluster)
        template_builder = ClusterTemplateBuilder()

        # If the cluster size is smaller than 9, take the size instead of 9 to avoid errors
        sample_size = min(len(cluster_gdf), 9)
        samples = cluster_gdf.sample(n=sample_size)

        for i, sample in samples.iterrows():
            summary_builder.add_sample(sample)

        forward_angle = ProbabilityHistogram(
            cluster_gdf["forward_angle"], "forward_angle", StreetInfluenceMap
        )
        forward_angle.add_influence(
            cluster_gdf["previous_segment_forward_angle"],
            "previous_segment_forward_angle",
        )

        forward_angle.add_influence(
            cluster_gdf["previous_segment_length"],
            "previous_segment_length",
        )
        forward_angle.add_influence(
            cluster_gdf["distance_to_last_intersection"],
            "distance_to_last_intersection",
        )

        segment_length = ProbabilityHistogram(
            cluster_gdf["length"], "segment_length", StreetInfluenceMap
        )
        segment_length.add_influence(
            cluster_gdf["previous_segment_forward_angle"],
            "previous_segment_forward_angle",
        )
        segment_length.add_influence(
            cluster_gdf["previous_segment_length"],
            "previous_segment_length",
        )
        segment_length.add_influence(
            cluster_gdf["distance_to_last_intersection"],
            "distance_to_last_intersection",
        )

        next_node_degree = ProbabilityHistogram(
            cluster_gdf["next_node_degree"], "next_node_degree", StreetInfluenceMap
        )
        next_node_degree.add_influence(
            cluster_gdf["previous_segment_forward_angle"],
            "previous_segment_forward_angle",
        )
        next_node_degree.add_influence(
            cluster_gdf["previous_segment_length"],
            "previous_segment_length",
        )
        next_node_degree.add_influence(
            cluster_gdf["distance_to_last_intersection"],
            "distance_to_last_intersection",
        )

        summary_builder.add_distribution(forward_angle)
        summary_builder.add_distribution(segment_length)
        summary_builder.add_distribution(next_node_degree)

        cluster = int(float(cluster))
        fp = output_dir / f"cluster_{cluster}.json"
        summary_builder.export_template(fp)
        # summary_builder.plot()
