from itertools import islice

import geopandas as gpd
import numpy as np
import polars as pl
from loguru import logger
from scipy.spatial import KDTree
from tqdm import tqdm

from config import cfg


def aggregate_building_neighborhoods(
    buildings: gpd.GeoDataFrame,
    max_distance_m: float = 200.0,
    kd_ball_radius: int = 300.0,
) -> gpd.GeoDataFrame:
    skipped = set()
    UNIQUE_ID = cfg.layers.buildings.unique_id
    buildings_pl = pl.from_pandas(buildings.drop(columns=["geometry"]))
    pl_idx_to_unique_id = buildings_pl[UNIQUE_ID].to_numpy()
    centroids = buildings.set_index(UNIQUE_ID).geometry.centroid
    buildings_pl = buildings_pl.with_columns(
        x=centroids.x.to_numpy(), y=centroids.y.to_numpy()
    )
    kdtree = KDTree(buildings_pl[["x", "y"]].to_numpy())

    CHUNK_SIZE = 100
    aggregations = {
        "area": ["mean", "std"],
        "building_group_enclosure_perimeter_coverage": ["mean", "std"],
        "shared_walls_ratio": ["mean", "std"],
        "shape_index": ["mean", "std"],
        "building_group_elongation_std": ["mean"],
        "building_group_area_std": ["mean"],
        "equivalent_rectangular_index": ["mean", "std"],
        "elongation": ["mean", "std"],
        "covered_area": ["mean", "std"],
        "alignment": ["mean", "std"],
        "squareness": ["mean", "std"],
        "approximate_height": ["mean", "std", "max"],
        "land_use_category": ["mode"],
    }

    for column in list(aggregations.keys()):
        if column not in buildings_pl.columns:
            del aggregations[column]
            skipped.add(column)

    custom_functions = {}
    new_columns = []
    radius_nb_type = f"radius_{kd_ball_radius:.0f}"
    for nb_type in ("enclosure", radius_nb_type):
        new_columns += [
            f"{column}::nb_{nb_type}::{method}"
            for column, methods in aggregations.items()
            for method in methods
        ] + list(custom_functions.keys())

    buildings_pl = buildings_pl.drop(new_columns)

    # Initialize new columns in road_edges_pl with nan
    for col in new_columns:
        if col not in buildings_pl.columns:
            buildings_pl = buildings_pl.with_columns(pl.lit(None).alias(col))

    pbar = tqdm(total=len(buildings))
    # the enclosure ids that occur in more than 200 buildings
    enormous_enclosures = buildings_pl.groupby(cfg.layers.enclosures.unique_id).agg(
        [pl.count(UNIQUE_ID).alias("bcount")]
    )
    enormous_enclosures = enormous_enclosures.filter(
        enormous_enclosures["bcount"] > 200
    )
    buildings_in_enormous_enclosures = enormous_enclosures.join(
        buildings_pl, on=cfg.layers.enclosures.unique_id, how="left"
    )
    buildings_in_enormous_enclosures = buildings_in_enormous_enclosures[
        UNIQUE_ID
    ].to_numpy()
    for i, building_chunk in enumerate(buildings_pl.iter_slices(CHUNK_SIZE)):
        offset = i * CHUNK_SIZE
        building_chunk: pl.DataFrame = building_chunk
        # buildings_neighbourhood_expansion = building_chunk.filter(
        #     ~pl.col(UNIQUE_ID).is_in(buildings_in_enormous_enclosures)
        # ).join(
        #     buildings_pl,
        #     on=cfg.layers.enclosures.unique_id,
        #     how="left",
        #     suffix="_right",
        # )
        # distance_x = (
        #     buildings_neighbourhood_expansion["x"]
        #     - buildings_neighbourhood_expansion["x_right"]
        # ).abs()
        # distance_y = (
        #     buildings_neighbourhood_expansion["y"]
        #     - buildings_neighbourhood_expansion["y_right"]
        # ).abs()
        # distance = (distance_x**2 + distance_y**2).sqrt()
        # buildings_neighbourhood_expansion = buildings_neighbourhood_expansion.filter(
        #     distance < max_distance_m
        # )
        #
        # result = buildings_neighbourhood_expansion.groupby(UNIQUE_ID).agg(
        #     [
        #         getattr(pl.col(column + "_right"), method)().alias(
        #             f"{column}::nb_enclosure::{method}"
        #         )
        #         for column, methods in aggregations.items()
        #         for method in methods
        #     ]
        #     + [
        #         pl.map_groups(exprs=args, function=func).alias(column)
        #         for column, (args, func) in custom_functions.items()
        #     ]
        # )
        #
        # buildings_pl = buildings_pl.update(result, on=UNIQUE_ID)

        buildings_kd_nb = kdtree.query_ball_point(
            building_chunk[["x", "y"]].to_numpy(), kd_ball_radius
        )
        buildings_kd_nb = {
            pl_idx_to_unique_id[idx + offset]: [
                pl_idx_to_unique_id[nb_idx] for nb_idx in neighbour_idxs
            ]
            for idx, neighbour_idxs in enumerate(buildings_kd_nb)
        }
        buildings_kd_expansion = pl.DataFrame(
            [
                (building_unique_id, nb_unique_id)
                for building_unique_id, neighbours in buildings_kd_nb.items()
                for nb_unique_id in neighbours
            ],
            schema=[UNIQUE_ID, "nb_unique_id"],
        )

        neighborhood_data = buildings_kd_expansion.join(
            buildings_pl, left_on="nb_unique_id", right_on=UNIQUE_ID, how="left"
        )

        result = neighborhood_data.groupby(UNIQUE_ID).agg(
            [
                getattr(pl.col(column), method)().alias(
                    f"{column}::nb_{radius_nb_type}::{method}"
                )
                for column, methods in aggregations.items()
                for method in methods
            ]
            # + [
            #     pl.map_groups(exprs=args, function=func).alias(column)
            #     for column, (args, func) in custom_functions.items()
            # ]
        )
        buildings_pl = buildings_pl.update(result, on=UNIQUE_ID)

        pbar.update(CHUNK_SIZE)

    pbar.close()

    buildings_pd = (
        buildings_pl.select([*new_columns, UNIQUE_ID]).to_pandas().set_index(UNIQUE_ID)
    )

    # Create column in edges if not exist
    for column in new_columns:
        if column not in buildings:
            buildings[column] = np.nan

    old_index_name = buildings.index.name
    buildings = buildings.set_index(UNIQUE_ID)
    buildings.update(buildings_pd, join="left", overwrite=True)
    if old_index_name is not None:
        buildings = buildings.set_index(old_index_name)
    else:
        buildings = buildings.reset_index()

    for column in skipped:
        logger.warning(
            f"Skipping column {column} because it does not exist in the data"
        )

    return buildings
