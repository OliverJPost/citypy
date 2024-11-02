import time
import traceback
from collections import deque
from itertools import islice
from typing import List

import geopandas as gpd
import networkx as nx
import numpy as np
import osmnx
import polars as pl
import rustworkx as rx
from loguru import logger
from tqdm import tqdm

from citypy.road_class import EdgeClass
from citypy.util.graph import graph_from_nodes_edges
from config import cfg

# def get_neighbor_edges(current_node, excluded_edge, input_graph):
#     return [
#         (node, neighbor)
#         for node, neighbor, edge_key in input_graph.edges(current_node, keys=True)
#         if (node, neighbor, edge_key) != excluded_edge
#         and (neighbor, node, edge_key) != excluded_edge
#     ]


def _find_edge_neighborhood(
    G: rx.PyDiGraph, start_edge, total_steps, grow_to_lower_class=False
):
    start_step = 0
    edges_to_check = deque([(start_edge, start_step)])
    inversed_edge = (start_edge[1], start_edge[0], start_edge[2])
    if G.has_edge(*inversed_edge[:2]):
        edges_to_check.append((inversed_edge, start_step))

    visited_edges = set()
    start_edge_class = get_edge_class(G, start_edge)

    while edges_to_check:
        edge, step = edges_to_check.popleft()
        if edge in visited_edges:
            continue
        visited_edges.add(edge)

        # Do not grow further if max steps reached
        if step == total_steps:
            continue
        neighbor_edges = get_neighbor_edges(G, edge)

        for neighbor_edge in neighbor_edges:
            if (
                not grow_to_lower_class
                and get_edge_class(G, neighbor_edge) < start_edge_class
            ):
                continue
            edges_to_check.append((neighbor_edge, step + 1))

    neighborhood_nodes = set([node for edge in visited_edges for node in edge[:2]])

    return list(visited_edges), list(neighborhood_nodes)


def find_edge_neighborhood(
    G: nx.Graph,
    start_edge: tuple,
    total_steps: int,
    grow_to_lower_class=False,
    grow_to_higher_class=False,
) -> tuple[List[tuple], List[int]]:
    start_step = 0
    edges_to_check = deque([(start_edge, start_step)])
    inversed_edge = (start_edge[1], start_edge[0], start_edge[2])
    if inversed_edge in G.edges(keys=True):
        edges_to_check.append((inversed_edge, start_step))

    visited_edges = set()
    start_edge_class = get_edge_class(G, start_edge)

    while edges_to_check:
        edge, step = edges_to_check.popleft()
        if edge in visited_edges:
            continue
        visited_edges.add(edge)

        # Do not grow further if max steps reached
        if step == total_steps:
            continue
        neighbor_edges = get_neighbor_edges(G, edge)

        for neighbor_edge in neighbor_edges:
            if (
                not grow_to_lower_class
                and get_edge_class(G, neighbor_edge) < start_edge_class
            ):
                continue
            if (
                not grow_to_higher_class
                and get_edge_class(G, neighbor_edge) > start_edge_class
            ):
                continue
            edges_to_check.append((neighbor_edge, step + 1))

    neighborhood_nodes = set([node for edge in visited_edges for node in edge[:2]])

    return list(visited_edges), list(neighborhood_nodes)


def get_neighbor_edges(G, edge):
    node1, node2, _ = edge
    node1_edges = set(G.edges(node1, keys=True))
    node2_edges = set(G.edges(node2, keys=True))
    return (node1_edges | node2_edges) - {edge}


def get_edge_class(G, edge):
    return EdgeClass.from_highway(G[edge[0]][edge[1]][edge[2]]["highway"])


def convert_nx_to_rx(nx_graph):
    node_mapping = {node: idx for idx, node in enumerate(nx_graph.nodes())}
    rx_graph = rx.PyDiGraph(multigraph=True)
    for node in node_mapping.values():
        rx_graph.add_node(node)

    for u, v, k in nx_graph.edges(keys=True):
        rx_graph.add_edge(node_mapping[u], node_mapping[v], k)

    return rx_graph, node_mapping


def uniform_orthogonality(bearings_and_lengths: list[pl.Series]):
    # TODO use https://appliednetsci.springeropen.com/articles/10.1007/s41109-019-0189-1

    bearings, lengths = bearings_and_lengths

    # bearings are in degrees
    if bearings.empty:
        return None

    bearings = bearings.reset_index(drop=True)
    lengths = lengths.reset_index(drop=True)

    sample_size = 20

    random_indices1 = np.random.choice(bearings.index, sample_size, replace=True)
    random_indices2 = np.random.choice(bearings.index, sample_size, replace=True)
    random_bearings1 = bearings.iloc[random_indices1].to_numpy()
    random_bearings2 = bearings.iloc[random_indices2].to_numpy()
    lengths1 = lengths.iloc[random_indices1].to_numpy()
    lengths2 = lengths.iloc[random_indices2].to_numpy()
    min_lengths = np.minimum(lengths1, lengths2)

    angle_differences = np.abs(random_bearings1 - random_bearings2)
    angle_differences = np.minimum(angle_differences, 360 - angle_differences)

    orthogonal_thresholds = [0, 90, 180, 270, 360]
    orthogonality_counts = sum(
        np.isclose(angle_differences, ortho_angle, atol=10)
        for ortho_angle in orthogonal_thresholds
    )

    # weighted by length
    orthogonality_counts_weighted = orthogonality_counts * min_lengths

    # Calculate orthogonality score as the fraction of orthogonal pairs
    orthogonality_score = orthogonality_counts_weighted.sum() / min_lengths.sum()

    return orthogonality_score  # , cardinality_score


def angle_shannon_entropy(args: List[pl.Series]):
    data = args[0]
    weight = args[1]
    bins = 35
    offset = 5
    hist, bin_edges = np.histogram(
        data + offset, weights=weight, bins=bins, range=(-180, 180)
    )
    hist = hist / hist.sum()
    hist = hist[hist > 0]
    return -np.sum(hist * np.log2(hist))


def bearing_cardinality(bearings: list[pl.Series]):
    bearings = bearings[0]
    # Check what percentage of the bearings are roughly around 0, 90, 180, 270 and 360
    cardinality_thresholds = [0, 90, 180, 270, 360]
    cardinality_counts = sum(
        np.isclose(bearings, cardinality_angle, atol=5)
        for cardinality_angle in cardinality_thresholds
    )
    return cardinality_counts.sum() / len(bearings)


def weighted_median(data_and_weights: list[pl.Series]):
    data, weights = data_and_weights
    df = pl.DataFrame({"value": data, "weight": weights})

    df_sorted = df.sort(by="value")
    cumulative_weights = df_sorted["weight"].cum_sum()
    total_weight = weights.sum()

    median_cutoff = total_weight / 2
    median_value = df_sorted.filter(cumulative_weights >= median_cutoff)["value"][0]

    return median_value


def weighted_mean(data_and_weights: list[pl.Series]):
    data, weights = data_and_weights
    # Discard NaN
    data_nan_idxs = data.is_null()
    weights_nan_idxs = weights.is_null()
    nan_idxs = data_nan_idxs | weights_nan_idxs
    data = data.filter(~nan_idxs)
    weights = weights.filter(~nan_idxs)

    if data.is_empty():
        return np.nan
    elif weights.is_empty():
        return data.mean()

    return (data * weights).sum() / weights.sum()


def weighted_percentage(data_and_weights: list[pl.Series]):
    data = data_and_weights[0]
    weights = data_and_weights[1]

    # Convert str to boolean
    if data.dtype == pl.Utf8:
        data = data.str.to_lowercase().map_dict({"true": True, "false": False})

    # Convert boolean data to integers (True -> 1, False -> 0)
    if data.dtype == pl.Boolean:
        data = data.cast(pl.Int32)

    return (data * weights).sum() / weights.sum()


def update(
    self: pl.DataFrame, df_other: pl.DataFrame, join_columns: list[str]
) -> pl.DataFrame:
    """Updates dataframe with the values in df_other after joining on the join_columns"""

    # The columns that will be updated
    columns = [c for c in df_other.columns if c not in join_columns]

    df_ans = (
        self.join(df_other, how="left", on=join_columns, suffix="_NEW")
        .with_columns(
            **{
                c: pl.coalesce([pl.col(c + "_NEW"), pl.col(c)])
                for c in columns  # <-This updates the columns
            }
        )
        .select(
            pl.all().exclude("^.*_NEW$")  # <- this drops the temporary '*_NEW' columns
        )
    )
    return df_ans


def aggregate_edge_neighborhoods(
    road_nodes: gpd.GeoDataFrame, road_edges: gpd.GeoDataFrame, steps: int
) -> gpd.GeoDataFrame:
    skipped = set()
    start_time = time.perf_counter()
    road_edges_pl = pl.from_pandas(road_edges.drop(columns=["geometry"]).reset_index())

    G = graph_from_nodes_edges(road_nodes, road_edges)
    for edge in G.edges(keys=True):
        # remove enclosure_ids attr if exists, prevents error in osmnx
        if "enclosure_ids" in G[edge[0]][edge[1]][edge[2]]:
            del G[edge[0]][edge[1]][edge[2]]["enclosure_ids"]
    road_section_graph = osmnx.simplify_graph(G)  # , track_merged=True)

    def chunks(iterator, size):
        iterator = iter(iterator)
        chunk = list(islice(iterator, size))
        while chunk:
            yield chunk
            chunk = list(islice(iterator, size))

    CHUNK_SIZE = 10000
    aggregations = {
        "stretch_linearity": ["mean", "std"],
        "forward_angle": ["mean", "std"],
        "continuity": ["mean", "std"],
        "stretch_curvilinearity": ["mean", "std"],
        "intersection_left_angle": ["mean", "std"],
        "right_neighbour_angle_deviation": ["mean", "std"],
        "right_neighbour_distance": ["mean", "std"],
        "forward_angle_abs": ["std"],
        "neighboring_street_orientation_deviation": ["mean", "std"],
        "section_length": ["mean", "std"],
    }
    # Remove any columns that are not in input
    for column in list(aggregations.keys()):
        if column not in road_edges_pl.columns:
            del aggregations[column]
            skipped.add(column)

    custom_functions = {
        "angle_entropy::nb": (
            ["bearing", "length", "type_category"],
            angle_shannon_entropy,
        ),
        "bearing_cardinality::nb": (["bearing"], bearing_cardinality),
        "dead_end_sections::nb::weighted_percentage": (
            ["dead_end_section", "length"],
            weighted_percentage,
        ),
        "continuity::nb::weighted_median": (["continuity", "length"], weighted_median),
        "stretch_curvilinearity::nb::weighted_mean": (
            ["stretch_curvilinearity", "length"],
            weighted_mean,
        ),
        "continuity::nb::weighted_mean": (
            ["continuity", "length"],
            weighted_mean,
        ),
        "right_neighbour_angle_deviation::nb::weighted_mean": (
            ["right_neighbour_angle_deviation", "length"],
            weighted_mean,
        ),
        "right_neighbour_distance::nb::weighted_mean": (
            ["right_neighbour_distance", "length"],
            weighted_mean,
        ),
        "forward_angle_abs::nb::weighted_mean": (
            ["forward_angle_abs", "length"],
            weighted_mean,
        ),
        "section_length::nb::weighted_mean": (
            ["section_length", "length"],
            weighted_mean,
        ),
        # "uniform_orthogonality::nb": (["bearing", "length"], uniform_orthogonality) FIXME pandas
    }
    # Remove any columns that are not in input
    for column_name, (dependent_columns, _) in list(custom_functions.items()):
        if not all(column in road_edges_pl.columns for column in dependent_columns):
            del custom_functions[column_name]
            skipped.add(column_name)

    new_columns = [
        f"{column}::nb::{method}"
        for column, methods in aggregations.items()
        for method in methods
    ] + list(custom_functions.keys())

    road_edges_pl = road_edges_pl.drop(new_columns)

    # Initialize new columns in road_edges_pl with nan
    for col in new_columns:
        if col not in road_edges_pl.columns:
            road_edges_pl = road_edges_pl.with_columns(pl.lit(None).alias(col))

    # fill na of column cfg.layers.road_edges.attributes.right_neighbour_distance with cfg.layers.road_edges.attributes.right_neighbour_max_distance_m
    try:
        road_edges_pl = road_edges_pl.with_columns(
            pl.coalesce(
                [
                    pl.col(cfg.layers.road_edges.attributes.right_neighbour_distance),
                    pl.lit(
                        cfg.layers.road_edges.attributes.right_neighbour_max_distance_m
                    ),
                ]
            ).alias(cfg.layers.road_edges.attributes.right_neighbour_distance)
        )
    except Exception as e:
        logger.error(e)
        stacktrace = traceback.format_exc()
        logger.error(stacktrace)
        logger.error(
            "Error filling na of column cfg.layers.road_edges.attributes.right_neighbour_distance with cfg.layers.road_edges.attributes.right_neighbour_max_distance_m"
        )

    pbar = tqdm(total=len(road_section_graph.edges))
    for section_chunk in chunks(
        road_section_graph.edges(keys=True, data="section_id"), CHUNK_SIZE
    ):
        section_neighborhoods = {
            section_id: find_edge_neighborhood(
                road_section_graph,
                tuple(section_edge),
                steps,
                grow_to_lower_class=False,
                grow_to_higher_class=False,
            )[0]
            for *section_edge, section_id in section_chunk
            if section_id
        }

        neighborhood_expansion = pl.DataFrame(
            [
                (
                    section_edge,
                    road_section_graph.get_edge_data(*neighborhood_edge).get(
                        "section_id"
                    ),
                )
                for section_edge, neighborhood_edges in section_neighborhoods.items()
                for neighborhood_edge in neighborhood_edges
            ],
            schema=["section_id", "neighborhood_section_id"],
        )

        neighborhood_data = neighborhood_expansion.join(
            road_edges_pl, left_on="neighborhood_section_id", right_on="section_id"
        )

        result = neighborhood_data.groupby("section_id").agg(
            [
                getattr(pl.col(column), method)().alias(f"{column}::nb::{method}")
                for column, methods in aggregations.items()
                for method in methods
            ]
            + [
                pl.map_groups(exprs=args, function=func).alias(column)
                for column, (args, func) in custom_functions.items()
            ]
        )

        road_edges_pl = road_edges_pl.update(result, on="section_id")
        # road_edges_pl = road_edges_pl.join(result, on="section_id")
        pbar.update(CHUNK_SIZE)

    pbar.close()

    edges_pd = (
        road_edges_pl.select([*new_columns, "u", "v", "key"])
        .to_pandas()
        .set_index(["u", "v", "key"])
    )

    # Create column in edges if not exist
    for column in new_columns:
        if column not in road_edges:
            road_edges[column] = np.nan

    road_edges.update(edges_pd, join="left", overwrite=True)

    for column in skipped:
        logger.warning(
            f"Skipped contextualize column: {column} because of missing source column."
        )

    return road_edges, None


def aggregate_major_edge_neighborhoods(
    road_nodes: gpd.GeoDataFrame, road_edges: gpd.GeoDataFrame, steps: int
) -> gpd.GeoDataFrame:
    skipped = set()
    start_time = time.perf_counter()
    major_road_edges = road_edges[
        road_edges["type_category"].isin(("highway", "major"))
    ]
    # Only where major_section_id is not null or na
    major_road_edges = major_road_edges[
        major_road_edges["major_section_id"].notnull()
        & major_road_edges["major_section_id"].notna()
    ]
    major_road_edges["major_section_id"] = major_road_edges["major_section_id"].astype(
        np.int64
    )
    road_edges_pl = pl.from_pandas(
        major_road_edges.drop(columns=["geometry"]).reset_index()
    )
    G = graph_from_nodes_edges(road_nodes, major_road_edges)

    for edge in G.edges(keys=True):
        # remove enclosure_ids attr if exists, prevents error in osmnx
        if "enclosure_ids" in G[edge[0]][edge[1]][edge[2]]:
            del G[edge[0]][edge[1]][edge[2]]["enclosure_ids"]
    road_section_graph = osmnx.simplify_graph(G)  # , track_merged=True)

    def chunks(iterator, size):
        iterator = iter(iterator)
        chunk = list(islice(iterator, size))
        while chunk:
            yield chunk
            chunk = list(islice(iterator, size))

    CHUNK_SIZE = 500
    aggregations = {
        "right_major_neighbour_angle_deviation": ["mean", "std"],
        "right_major_neighbour_distance": ["mean", "std"],
        "major_section_length": ["mean", "std"],
    }
    # Remove any columns that are not in input
    for column in list(aggregations.keys()):
        if column not in road_edges_pl.columns:
            del aggregations[column]
            skipped.add(column)

    custom_functions = {
        "angle_entropy::nb_major": (
            ["bearing", "length", "type_category"],
            angle_shannon_entropy,
        ),
        "bearing_cardinality::nb_major": (["bearing"], bearing_cardinality),
        "continuity::nb_major::weighted_median": (
            ["continuity", "length"],
            weighted_median,
        ),
        "stretch_curvilinearity::nb_major::weighted_mean": (
            ["stretch_curvilinearity", "length"],
            weighted_mean,
        ),
        "continuity::nb_major::weighted_mean": (
            ["continuity", "length"],
            weighted_mean,
        ),
        "right_major_neighbour_angle_deviation::nb_major::weighted_mean": (
            ["right_major_neighbour_angle_deviation", "length"],
            weighted_mean,
        ),
        "right_major_neighbour_distance::nb_major::weighted_mean": (
            ["right_major_neighbour_distance", "length"],
            weighted_mean,
        ),
        "forward_angle_abs::nb_major::weighted_mean": (
            ["forward_angle_abs", "length"],
            weighted_mean,
        ),
    }
    # Remove any columns that are not in input
    for column_name, (dependent_columns, _) in list(custom_functions.items()):
        if not all(column in road_edges_pl.columns for column in dependent_columns):
            del custom_functions[column_name]
            skipped.add(column_name)

    new_columns = [
        f"{column}::nb_major::{method}"
        for column, methods in aggregations.items()
        for method in methods
    ] + list(custom_functions.keys())

    road_edges_pl = road_edges_pl.drop(new_columns)

    # Initialize new columns in road_edges_pl with nan
    for col in new_columns:
        if col not in road_edges_pl.columns:
            road_edges_pl = road_edges_pl.with_columns(pl.lit(None).alias(col))

    # fill na of column cfg.layers.road_edges.attributes.right_neighbour_distance with cfg.layers.road_edges.attributes.right_neighbour_max_distance_m
    try:
        road_edges_pl = road_edges_pl.with_columns(
            pl.coalesce(
                [
                    pl.col(cfg.layers.road_edges.attributes.right_neighbour_distance),
                    pl.lit(
                        cfg.layers.road_edges.attributes.right_neighbour_max_distance_m
                    ),
                ]
            ).alias(cfg.layers.road_edges.attributes.right_neighbour_distance)
        )
    except Exception as e:
        logger.error(e)
        stacktrace = traceback.format_exc()
        logger.error(stacktrace)
        logger.error(
            "Error filling na of column cfg.layers.road_edges.attributes.right_neighbour_distance with cfg.layers.road_edges.attributes.right_neighbour_max_distance_m"
        )

    pbar = tqdm(total=len(road_section_graph.edges))
    for section_chunk in chunks(
        road_section_graph.edges(keys=True, data="major_section_id"), CHUNK_SIZE
    ):
        section_neighborhoods = {
            tuple([section_ids])
            if isinstance(section_ids, int)
            else tuple(section_ids): find_edge_neighborhood(
                road_section_graph,
                tuple(section_edge),
                steps,
                grow_to_lower_class=False,
                grow_to_higher_class=True,
            )[0]
            for *section_edge, section_ids in section_chunk
            if section_ids
        }

        df_data = []
        for section_edges, neighborhood_edges in section_neighborhoods.items():
            for section_edge in section_edges:
                for neighborhood_edge in neighborhood_edges:
                    neighborhood_edge_id = road_section_graph.get_edge_data(
                        *neighborhood_edge
                    ).get("major_section_id")
                    if isinstance(neighborhood_edge_id, int):
                        df_data.append((section_edge, neighborhood_edge_id))
                    else:
                        for nb_id in neighborhood_edge_id:
                            df_data.append((section_edge, nb_id))

        neighborhood_expansion = pl.DataFrame(
            df_data,
            schema=[
                ("major_section_id", pl.datatypes.Int64),
                ("neighborhood_major_section_id", pl.datatypes.Int64),
            ],
        )

        neighborhood_data = neighborhood_expansion.join(
            road_edges_pl,
            left_on="neighborhood_major_section_id",
            right_on="major_section_id",
        )

        result = neighborhood_data.groupby("major_section_id").agg(
            [
                getattr(pl.col(column), method)().alias(f"{column}::nb_major::{method}")
                for column, methods in aggregations.items()
                for method in methods
            ]
            + [
                pl.map_groups(exprs=args, function=func).alias(column)
                for column, (args, func) in custom_functions.items()
            ]
        )

        road_edges_pl = road_edges_pl.update(result, on="major_section_id")
        # road_edges_pl = road_edges_pl.join(result, on="section_id")
        pbar.update(CHUNK_SIZE)

    pbar.close()

    edges_pd = (
        road_edges_pl.select([*new_columns, "u", "v", "key"])
        .to_pandas()
        .set_index(["u", "v", "key"])
    )

    # Create column in edges if not exist
    for column in new_columns:
        if column not in road_edges:
            road_edges.loc[column] = np.nan

    road_edges.update(edges_pd, join="left", overwrite=True)

    for column in skipped:
        logger.warning(
            f"Skipped contextualize column: {column} because of missing source column."
        )

    return road_edges, None
