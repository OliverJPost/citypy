import traceback
from enum import Enum
from functools import partial
from pathlib import Path
from typing import Callable, List, Optional

import geopandas as gpd
import libpysal
import momepy
import pyogrio.errors
import typer
from libpysal import graph
from loguru import logger

from citypy.cli import app
from citypy.console import console
from citypy.process import layers as process_layers
from citypy.process import metrics
from citypy.process.layers.plots import PlotSeeds
from citypy.util.gpkg import GeoPackage, LockReasons
from citypy.util.graph import convert_graph_to_gdfs, graph_from_nodes_edges
from config import cfg

TESSELATION = cfg.layers.tesselation.name
ROAD_EDGES = cfg.layers.road_edges.name
ROAD_NODES = cfg.layers.road_nodes.name
BUILDINGS = cfg.layers.buildings.name
BLOCKS = cfg.layers.blocks.name
LANDUSE = cfg.layers.landuse.name
ENCLOSURES = cfg.layers.enclosures.name
EXTENTS = cfg.layers.extents.name
CITY_CENTER = cfg.layers.city_center.name


class ProcessErrors(Enum):
    LOCKED_GPKG = 25


@app.command()
def process(
    input_gpkg: Path = typer.Argument(
        help="Path to the input GPKG file.",
        exists=True,
    ),
    select: List[str] = typer.Option(
        [],
        "--select",
        "-s",
        help="Select which metrics and layers to generate. If not given, all are generated.",
    ),
    overwrite: bool = typer.Option(
        False,
        "--overwrite",
        "-o",
        help="Overwrite existing columns and layers.",
    ),
    crash_early: bool = typer.Option(
        False,
        "--crash-early",
        help="Crash early if an error occurs.",
    ),
) -> None:
    gpkg = GeoPackage(input_gpkg, verbose=True)

    # If the download phase was not completed, do not process this file
    if gpkg.is_locked() and gpkg.lock_reason() == LockReasons.DOWNLOAD:
        logger.critical(gpkg.lock_reason())
        exit(ProcessErrors.LOCKED_GPKG)

    gpkg.lock(LockReasons.PROCESS)

    layers = {
        "road_edges": gpkg.read_vector(ROAD_EDGES).set_index(["u", "v", "key"]),
        "road_nodes": gpkg.read_vector(ROAD_NODES),
        "buildings": gpkg.read_vector(BUILDINGS),
        "extents": gpkg.read_vector(EXTENTS),
        "city_center": gpkg.read_vector(CITY_CENTER),
        "landuse": gpkg.read_vector(LANDUSE),
    }
    layers["road_graph"] = LazyRoadGraph(layers["road_nodes"], layers["road_edges"])

    all_failed = set()
    all_modified = set()

    layers, failed, modified = generate_column_metrics(
        layers, overwrite, select, metrics.before_layers_metrics, crash_early
    )
    all_failed |= failed
    all_modified |= modified

    layers = add_layers(layers, gpkg, select, overwrite)

    layers, failed, modified = generate_column_metrics(
        layers, overwrite, select, metrics.before_plots_metrics, crash_early
    )
    all_failed |= failed
    all_modified |= modified

    ps = PlotSeeds()
    if ps.should_generate(gpkg, select, overwrite):
        layers.update(
            ps.generate(
                road_edges=layers["road_edges"],
                buildings=layers["buildings"],
                tesselation=layers["tesselation"],
            ),
        )
    else:
        layers["plots"] = None  # FIXME
    layers["spatial_weights"] = LazySpatialWeights(layers["buildings"])

    layers, failed, modified = generate_graph_metrics(
        layers, overwrite, select, crash_early
    )
    all_failed |= failed
    all_modified |= modified
    # Checkpoint
    export_layers(gpkg, layers, all_modified)

    layers, failed, modified = generate_column_metrics(
        layers, overwrite, select, metrics.all_column_metrics, crash_early
    )
    all_failed |= failed
    all_modified |= modified

    export_layers(gpkg, layers, all_modified)

    gpkg.unlock()
    for metric in all_failed:
        logger.warning(f"Failed to calculate metric: {metric}. See stack trace above.")


def add_layers(layers, gpkg, select, overwrite):
    all_modified_layers = set()

    for layer in process_layers.all_layers:
        if layer.should_generate(gpkg, select, overwrite):
            modified_layers = layer.generate(**layers)
            layers.update(modified_layers)
            all_modified_layers.update(modified_layers.keys())
        else:
            try:
                layers[layer.layer_name] = gpkg.read_vector(layer.layer_name)
            except pyogrio.errors.DataLayerError:
                console.log(
                    f"Layer [bold]{layer.layer_name}[/bold] not found in GPKG. Skipping..."
                )

    for layer_name in all_modified_layers:
        gpkg.write_vector(layers[layer_name], getattr(cfg.layers, layer_name).name)

    return layers


def generate_column_metrics(layers, overwrite, select, list_of_metrics, crash_early):
    failed = set()
    modified = set()
    for layer_name, metric in list_of_metrics:
        if metric.should_calculate(layers[layer_name], select, overwrite):
            layer = layers[layer_name]
            try:
                value = metric.calculate(**layers)
                layer[metric.attr_name] = value
                modified.add(layer_name)
            except Exception as e:
                if crash_early:
                    raise e
                logger.error(f"Error calculating {metric.attr_name}: {e}")
                # print whole stacktracke
                traceback.print_exc()
                failed.add(metric.__class__.__name__)

    return layers, failed, modified


def generate_graph_metrics(layers, overwrite, select, crash_early):
    failed = set()
    modified = False
    for layer_name, metric in metrics.all_graph_metrics:
        if metric.should_calculate(layers[layer_name], select, overwrite):
            try:
                layers["road_graph"] = metric.calculate(**layers)
                modified = True
            except Exception as e:
                if crash_early:
                    raise e
                logger.error(f"Error calculating {metric.attr_name}: {e}")
                traceback.print_exc()
                failed.add(metric.__class__.__name__)

    if modified:
        road_nodes, road_edges = convert_graph_to_gdfs(layers["road_graph"])
        layers["road_edges"] = road_edges
        layers["road_nodes"] = road_nodes

    modified_set = {"road_graph"} if modified else set()
    return layers, failed, modified_set


class LazySpatialWeights:
    def __init__(self, buildings: gpd.GeoDataFrame, verbose=True):
        self.buildings = buildings
        self.verbose = verbose
        self._weights = None

    def __getattr__(self, item):
        if item in self.__dict__:
            return self.__dict__[item]

        if self._weights is None:
            if self.verbose:
                logger.info("Generating spatial weights...")
            # tessellation = momepy.Tessellation(
            #     self.buildings,
            #     unique_id="id::building",
            #     limit=self.extents.to_crs(self.buildings.crs).union_all(),
            #     segment=3.0,  # Very high, so tesselation is inaccurate but that's okay
            #     verbose=True,
            # )
            #
            # tesselation = tessellation.tessellation
            # tesselation_without_multipolygons = tesselation[~tesselation.geometry.has_z]
            # contiguity = graph.Graph.build_contiguity(
            #     tesselation_without_multipolygons, rook=False
            # )
            contiguity = graph.Graph.build_knn(self.buildings.centroid, k=20)

            self._weights = contiguity

        return getattr(self._weights, item)


class LazyRoadGraph:
    def __init__(self, nodes: gpd.GeoDataFrame, edges: gpd.GeoDataFrame, verbose=True):
        self._nodes = nodes
        self._edges = edges
        self.verbose = verbose
        self._graph = None

    def __getattr__(self, item):
        if item in self.__dict__:
            return self.__dict__[item]

        if self._graph is None:
            self.generate()

        return getattr(self._graph, item)

    def __getitem__(self, item):
        if self._graph is None:
            self.generate()

        return self._graph[item]

    def __setitem__(self, key, value):
        if self._graph is None:
            self.generate()

        self._graph[key] = value

    def __delitem__(self, key):
        if self._graph is None:
            self.generate()

        del self._graph[key]

    def __iter__(self):
        if self._graph is None:
            self.generate()

        return iter(self._graph)

    def __len__(self):
        if self._graph is None:
            self.generate()

        return len(self._graph)

    def __contains__(self, item):
        if self._graph is None:
            self.generate()

        return item in self._graph

    def generate(self):
        if self.verbose:
            logger.info("Generating lazy road graph...")
        self._graph = graph_from_nodes_edges(self._nodes, self._edges)
        if self.verbose:
            logger.info("Road graph generated! Continuing to metric.")


class LazyHighSpatialWeights:
    def __init__(self, original_weights, verbose=True):
        self.original_weights = original_weights
        self.verbose = verbose
        self._high_weights = None

    def __getattr__(self, item):
        if self._high_weights is None:
            if self.verbose:
                print("Generating high spatial weights...")
            self._high_weights = momepy.sw_high(k=3, weights=self.original_weights)
        return getattr(self._high_weights, item)


def export_layers(gpkg, layers, modified):
    if "buildings" in modified:
        gpkg.write_vector(layers["buildings"], BUILDINGS)
    if "road_edges" in modified or "road_nodes" in modified or "road_graph" in modified:
        gpkg.write_vector(layers["road_edges"], ROAD_EDGES)
        gpkg.write_vector(layers["road_nodes"], ROAD_NODES)
    if layers["tesselation"] is not None and "tesselation" in modified:
        gpkg.write_vector(layers["tesselation"], TESSELATION)
    if layers["enclosures"] is not None and "enclosures" in modified:
        gpkg.write_vector(layers["enclosures"], ENCLOSURES)
    if layers["plots"] is not None and "plots" in modified:
        gpkg.write_vector(layers["plots"], "plots")
