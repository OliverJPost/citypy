import time
import traceback
from enum import Enum
from typing import Optional

import pandas as pd
import typer
from loguru import logger

from citypy.cli import app
from citypy.contextualize.building_neighborhood import aggregate_building_neighborhoods
from citypy.contextualize.merge import (
    merge_buildings_with_partition_attributes,
    # merge_blocks_with_building_attributes,
    merge_edges_with_partition_attributes,
)
from citypy.contextualize.road_neighborhood import (
    aggregate_edge_neighborhoods,
    aggregate_major_edge_neighborhoods,
)
from citypy.util.gpkg import GeoPackage, LockReasons
from citypy.util.graph import graph_from_nodes_edges
from config import cfg


class ContextualizeErrors(Enum):
    LOCKED_GPKG = 1


@app.command()
def contextualize(
    input_gpkg: str = typer.Argument(
        help="Path to the input GPKG file.",
    ),
    buildings: Optional[bool] = typer.Option(
        False,
        help="Whether to contextualize buildings.",
    ),
    regular: Optional[bool] = typer.Option(
        False,
        help="Whether to contextualize regular edges.",
    ),
    major: Optional[bool] = typer.Option(
        False,
        help="Whether to contextualize major edges.",
    ),
) -> None:
    start_time = time.time()
    input_gpkg = input_gpkg

    gpkg = GeoPackage(input_gpkg, verbose=True)
    if gpkg.is_locked() and gpkg.lock_reason() in (
        LockReasons.DOWNLOAD,
        LockReasons.PROCESS,
    ):
        logger.critical(gpkg.lock_reason())
        exit(ContextualizeErrors.LOCKED_GPKG)
    gpkg.lock(LockReasons.CONTEXTUALIZE)

    if buildings:
        enclosures = gpkg.read_vector(cfg.layers.enclosures.name)
        buildings = gpkg.read_vector(cfg.layers.buildings.name)
        # blocks = gpkg.read_vector(cfg.layers.blocks.name)

        # road_edges = merge_edges_with_partition_attributes(
        #     road_edges, buildings, enclosures
        # )
        logger.info("Merging buildings with partition attributes...")
        try:
            buildings = merge_buildings_with_partition_attributes(buildings, enclosures)
        except Exception as e:
            logger.error(e)
            stacktrace = traceback.format_exc()
            logger.error(stacktrace)
            logger.error("Error merging buildings with partition attributes")

        logger.info("Aggregating building neighborhoods...")
        buildings = aggregate_building_neighborhoods(buildings)
        gpkg.write_vector(buildings, cfg.layers.buildings.name)

        # Memory clean up
        del buildings
        del enclosures

    road_edges = gpkg.read_vector(cfg.layers.road_edges.name).set_index(
        ["u", "v", "key"]
    )
    road_edges = road_edges.sort_index()
    road_nodes = gpkg.read_vector(cfg.layers.road_nodes.name)

    if regular:
        logger.info("Aggregating edge neighborhoods...")
        road_edges, association_table = aggregate_edge_neighborhoods(
            road_nodes, road_edges, steps=5
        )
        gpkg.write_vector(road_edges, cfg.layers.road_edges.name)
    if major:
        logger.info("Aggregating major edge neighborhoods...")
        road_edges, association_table = aggregate_major_edge_neighborhoods(
            road_nodes, road_edges, steps=7
        )
        # remove items with index that are strings
        road_edges = road_edges[
            ~road_edges.index.get_level_values(0).astype(str).str.contains("nb")
        ]
        road_edges.index = pd.MultiIndex.from_tuples(road_edges.index)
        road_edges["u"] = road_edges.index.get_level_values(0)
        road_edges["v"] = road_edges.index.get_level_values(1)
        road_edges["key"] = road_edges.index.get_level_values(2)
        road_edges = road_edges.reset_index(drop=True)
        gpkg.write_vector(road_edges, cfg.layers.road_edges.name)

    gpkg.unlock()

    print(f"Contextualization took {time.time() - start_time:.2f} seconds.")
