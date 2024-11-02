import geopandas as gpd
import networkx as nx
import osmnx
from loguru import logger
from shapely import Polygon

from citypy.console import conditional_status
from citypy.util.graph import convert_graph_to_gdfs
from config import cfg

ESSENTIAL_ROAD_NODE_COLUMNS = ["geometry", cfg.layers.road_nodes.unique_id, "x", "y"]

ESSENTIAL_ROAD_EDGE_COLUMNS = [
    "geometry",
    "u",
    "v",
    "key",
    cfg.layers.road_edges.unique_id,
    "highway",
    "length",
    "tunnel",
    "width",
    "bridge",
    "junction",
    cfg.layers.road_edges.attributes.type_category,
]


def download_raw_osm_road_graph(extents: Polygon) -> nx.MultiDiGraph:
    return osmnx.graph_from_polygon(extents, network_type="all", simplify=False)


def add_unique_id(gdf: gpd.GeoDataFrame, id_column: str) -> gpd.GeoDataFrame:
    gdf[id_column] = range(len(gdf))
    return gdf


def filter_essential_columns(road_nodes, road_edges):
    return road_nodes.filter(ESSENTIAL_ROAD_NODE_COLUMNS), road_edges.filter(
        ESSENTIAL_ROAD_EDGE_COLUMNS
    )


TYPE_CATEGORY_MAPPING = {
    "highway": ("motorway", "motorway_link", "trunk", "trunk_link"),
    "major": ("primary", "secondary", "tertiary"),
    "street": ("living_street", "residential", "pedestrian", "unclassified", "service"),
    "path": ("path", "footway", "steps"),
}


def add_simplified_type_category(road_nodes):
    road_nodes[cfg.layers.road_edges.attributes.type_category] = road_nodes[
        "highway"
    ].map(lambda x: next((k for k, v in TYPE_CATEGORY_MAPPING.items() if x in v), None))

    return road_nodes


def get_road_data(
    extents: Polygon, local_crs, essentials_only=True, verbose=True
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    with conditional_status(verbose) as status:
        status.update(f"Downloading OSM road graph...")
        logger.info("Downloading OSM road graph...")
        graph = download_raw_osm_road_graph(extents)
        status.update(f"Converting road graph to GeoDataFrames...")
        logger.info("Converting road graph to GeoDataFrames...")
        graph = osmnx.project_graph(graph, to_crs=local_crs)
        road_nodes, road_edges = convert_graph_to_gdfs(graph)
        # road_nodes = road_nodes.to_crs(road_nodes.estimate_utm_crs())
        # road_edges = road_edges.to_crs(road_edges.estimate_utm_crs())
        road_nodes = add_unique_id(road_nodes, cfg.layers.road_nodes.unique_id)
        road_edges = add_unique_id(road_edges, cfg.layers.road_edges.unique_id)
        road_edges = add_simplified_type_category(road_edges)
        if essentials_only:
            road_nodes, road_edges = filter_essential_columns(road_nodes, road_edges)
    return road_nodes, road_edges
