import ast
from collections import defaultdict

import geopandas as gpd

from config import cfg

# def merge_blocks_with_building_attributes(
#     blocks: gpd.GeoDataFrame, buildings: gpd.GeoDataFrame
# ) -> gpd.GeoDataFrame:
#     blocks = blocks.set_index("block_id") if "block_id" in blocks.columns else blocks
#     block_id_to_buildings = buildings.groupby("block_id")
#
#     joined = defaultdict(dict)
#
#     for block_id, buildings_group in block_id_to_buildings:
#         joined["buildings::count"][block_id] = len(buildings_group)
#         bl = cfg.layers.buildings.attributes
#         joined["buildings::area::mean"][block_id] = buildings_group[bl.area].mean()
#         joined["buildings::area::std"][block_id] = buildings_group[bl.area].std()
#         joined["buildings::shared_walls_ratio"][block_id] = buildings_group[
#             bl.shared_walls_ratio
#         ].mean()
#         joined["buildings::covered_area::mean"][block_id] = buildings_group[
#             bl.covered_area
#         ].mean()
#         joined["buildings::mean_interbuilding_distance::mean"][block_id] = (
#             buildings_group[bl.mean_interbuilding_distance].mean()
#         )
#
#     for attribute_name, attributes_values in joined.items():
#         blocks[attribute_name] = blocks.index.map(attributes_values)
#
#     return blocks


def merge_edges_with_partition_attributes(
    road_edges: gpd.GeoDataFrame,
    buildings: gpd.GeoDataFrame,
    # blocks: gpd.GeoDataFrame,
    enclosures: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    edges = road_edges
    if "edge_id" in edges.columns:
        edges.set_index("edge_id", inplace=True)
    edge_to_buildings = buildings.groupby("closest_road_edge_id")
    # if "block_id" in buildings.columns:
    #     blocks = blocks.set_index("block_id")

    joined = defaultdict(dict)

    # bl = cfg.layers.blocks.attributes
    # block_joins = {
    #     "block::circular_compactness": (["mean"], bl.circular_compactness),
    #     "block::equivalent:rectangular_index": (
    #         ["mean"],
    #         bl.equivalent_rectangular_index,
    #     ),
    #     "block::rectangularity": (["mean"], bl.rectangularity),
    # }
    # for edge_id, buildings_group in edge_to_buildings:
    #     joined["buildings::count"][edge_id] = len(buildings_group)
    #     unique_block_ids = buildings_group["block_id"].dropna().unique()
    #     joined["blocks::count"][edge_id] = len(unique_block_ids)
    #     if len(unique_block_ids) == 0:
    #         continue
    #
    #     relevant_blocks = blocks.loc[unique_block_ids]
    #     for joined_attribute_name, (methods, attribute_name) in block_joins.items():
    #         for method in methods:
    #             result = getattr(relevant_blocks[attribute_name], method)()
    #             joined[joined_attribute_name][edge_id] = result

    el = cfg.layers.enclosures.attributes
    enclosure_joins = {
        "enclosures::rectangularity": (["mean"], el.rectangularity),
        "enclosures::elongation": (["mean"], el.elongation),
        "enclosures::fractal_dimension": (["mean"], el.fractal_dimension),
        "enclosures::circular_compactness": (["mean"], el.circular_compactness),
    }

    enclosure_ids = road_edges.enclosure_ids.apply(
        lambda x: ast.literal_eval(x) if x else ()
    )
    for edge_id in edges.index:
        relevant_enclosures = enclosures.loc[enclosure_ids.loc[edge_id]]
        for joined_attribute_name, (methods, attribute_name) in enclosure_joins.items():
            for method in methods:
                result = getattr(relevant_enclosures[attribute_name], method)()
                joined[f"{joined_attribute_name}::{method}"][edge_id] = result

    for joined_attribute in joined:
        if isinstance(joined[joined_attribute], dict):
            edges[joined_attribute] = edges.index.map(joined[joined_attribute])
        else:
            edges[joined_attribute] = joined[joined_attribute]

    # back to u, v, key
    edges.set_index(["u", "v", "key"], inplace=True)

    return edges


def perform_aggregation_joins(join_dict, target_df, source_df, joined):
    for joined_attribute_name, (methods, attribute_name) in join_dict.items():
        for method in methods:
            # Call method of df of aggregation method i.e. df.mean()
            result = getattr(source_df[attribute_name], method)()
            joined[joined_attribute_name + f"::{method}"] = result.reindex(
                target_df.index, fill_value=None
            )

    return joined


def merge_buildings_with_partition_attributes(
    buildings, enclosures
) -> gpd.GeoDataFrame:
    enclosure_id = buildings[cfg.layers.enclosures.unique_id]

    buildings[cfg.layers.enclosures.attributes.enclosurue_building_area_ratio] = (
        enclosure_id.map(
            enclosures.set_index(cfg.layers.enclosures.unique_id)[
                cfg.layers.enclosures.attributes.enclosurue_building_area_ratio
            ]
        )
    )

    return buildings
