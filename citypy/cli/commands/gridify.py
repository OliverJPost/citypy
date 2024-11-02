from pathlib import Path
from typing import Annotated, List

import geopandas as gpd
import numpy as np
import pandas as pd
import scipy
import typer
from loguru import logger
from pyogrio.errors import DataLayerError
from scipy import ndimage
from shapely import Polygon
from tqdm import tqdm

from citypy.cli import app
from config import cfg


def create_grid(bounds, width=100, height=100):
    logger.info(f"Creating grid for bounds {bounds}")
    minx, miny, maxx, maxy = bounds
    rows = int((maxy - miny) / height)
    cols = int((maxx - minx) / width)
    polygons = []
    for i in range(cols):
        for j in range(rows):
            polygons.append(
                Polygon(
                    [
                        (minx + i * width, miny + j * height),
                        (minx + (i + 1) * width, miny + j * height),
                        (minx + (i + 1) * width, miny + (j + 1) * height),
                        (minx + i * width, miny + (j + 1) * height),
                    ]
                )
            )
    return polygons


def grid_to_numpy(grid, attribute):
    values = grid[attribute]
    # map unique strings to ids
    unique_values = values.unique()
    value_map = {
        value: value for idx, value in enumerate(unique_values) if value is not None
    }
    na_value = 99
    values = values.fillna(na_value).to_numpy()

    bounds = grid.total_bounds
    width, height = (
        grid.geometry.apply(lambda geom: geom.bounds[2] - geom.bounds[0]).mean(),
        grid.geometry.apply(lambda geom: geom.bounds[3] - geom.bounds[1]).mean(),
    )
    rows = int((bounds[3] - bounds[1]) / height)
    cols = int((bounds[2] - bounds[0]) / width)

    assert (
        len(values) == rows * cols
    ), f"Length of values {len(values)} does not match grid size {rows}x{cols}"

    grid_np = values.reshape((cols, rows))
    return grid_np


def apply_np_to_grid(grid, np_grid, attribute):
    values = np_grid.flatten()
    grid[attribute] = values
    return grid


@app.command()
def gridify(
    input_gpkg: Annotated[
        Path,
        typer.Argument(
            help="Path to the input GPKG file.",
            exists=True,
        ),
    ],
    layer_name: Annotated[
        str,
        typer.Argument(
            help="Name of the layer to plot clusters.",
        ),
    ],
    mode_columns: Annotated[
        List[str],
        typer.Option(
            ...,
            "--mode-columns",
            "--mode",
            help="Name of the columns to calculate the mode and add to the grid.",
        ),
    ] = None,
    most_intersecting_columns: Annotated[
        List[str],
        typer.Option(
            ...,
            "--most-intersecting-columns",
            "--most-intersecting",
            help="Name of the columns to calculate the most intersecting and add to the grid.",
        ),
    ] = None,
    mean_columns: Annotated[
        List[str],
        typer.Option(
            ...,
            "--mean-columns",
            "--mean",
            help="Name of the columns to calculate the mean and add to the grid.",
        ),
    ] = None,
    sum_columns: Annotated[
        List[str],
        typer.Option(
            ...,
            "--sum-columns",
            "--sum",
            help="Name of the columns to calculate the sum and add to the grid.",
        ),
    ] = None,
    count_columns: Annotated[
        List[str],
        typer.Option(
            ...,
            "--count-columns",
            "--count",
            help="Name of the columns to calculate the count and add to the grid.",
        ),
    ] = None,
    overwrite: bool = typer.Option(
        False,
        "--overwrite",
        "-o",
        help="Overwrite the grid layer if it already exists.",
    ),
) -> None:
    input_gpkg = input_gpkg
    gdf = gpd.read_file(input_gpkg, layer=layer_name, engine="pyogrio")
    gdf.to_crs(gdf.estimate_utm_crs(), inplace=True)  # TODO test in a different way
    # test if grid already exists in gpkg
    try:
        if overwrite:
            raise DataLayerError  # FIXME: This is a hack to force the except block to run
        grid_gdf = gpd.read_file(
            input_gpkg, layer=cfg.layers.grid.name, engine="pyogrio"
        )
    except DataLayerError:
        logger.info("Creating grid...")
        extents = gpd.read_file(
            input_gpkg, layer=cfg.layers.extents.name, engine="pyogrio"
        )
        extents = extents.to_crs(gdf.crs)
        bounds = extents.total_bounds
        polygons = create_grid(bounds, width=100, height=100)
        grid_gdf = gpd.GeoDataFrame(pd.DataFrame({"geometry": polygons}))
        logger.info(f"Created grid with {len(grid_gdf)} cells")
        logger.info("Distance to city center")

        city_center = gpd.read_file(
            input_gpkg, layer=cfg.layers.city_center.name, engine="pyogrio"
        ).geometry.unary_union.centroid
        grid_gdf["distance_to_city_center"] = grid_gdf.geometry.centroid.distance(
            city_center
        )
        #
        # logger.info("Distance to highways")
        # all_roads = gpd.read_file(
        #     input_gpkg, layer=cfg.layers.road_edges.name, engine="pyogrio"
        # )
        # highways = all_roads[all_roads["type_category"] == "highway"]
        # #
        # # grid_gdf["distance_to_highway"] = grid_gdf.geometry.apply(
        # #     lambda x: highways.distance(x).min()
        # # )
        # grid_gdf["contains_highway"] = grid_gdf.geometry.apply(
        #     lambda x: highways.intersects(x).any()
        # )
        # #
        # # logger.info("Distance to primary")
        # major = all_roads[all_roads["type_category"] == "major"]
        # #
        # grid_gdf["contains_major"] = grid_gdf.geometry.apply(
        #     lambda x: major.intersects(x).any()
        # )
        #
        # logger.info("Water percentage")
        # water = gpd.read_file(input_gpkg, layer=cfg.layers.water.name, engine="pyogrio")
        # grid_gdf["water_percentage"] = grid_gdf.geometry.apply(
        #     lambda x: water.intersection(x).area.sum() / x.area
        # )
    #
    # def process_chunk(chunk):
    #     # Intersect chunk geometries with the whole gdf and compute the intersection areas
    #     intersections = gpd.sjoin(gdf, chunk, how="inner", predicate="intersects")
    #     intersection_areas = intersections.geometry.intersection(chunk.geometry).area
    #
    #     # Compute area intersection in a vectorised manner
    #     chunk["area"] = chunk.geometry.apply(
    #         lambda geom: gdf[gdf.intersects(geom)].geometry.intersection(geom).area
    #     )
    #
    #     for column in mode_columns:
    #         column = column.strip()
    #         chunk[f"{layer_name}::{column}::mode"] = chunk["intersecting"].apply(
    #             lambda intersecting_geometries: intersecting_geometries[column]
    #             .mode()
    #             .iloc[0]
    #             if not intersecting_geometries[column].mode().empty
    #             else None
    #         )
    #
    #     for column in most_intersecting_columns:
    #         column = column.strip()
    #         chunk[f"{layer_name}::{column}::most_intersecting"] = chunk[
    #             "intersecting"
    #         ].apply(
    #             lambda intersecting_geometries: intersecting_geometries.groupby(column)[
    #                 "area"
    #             ]
    #             .sum()
    #             .idxmax()
    #             if column in intersecting_geometries.columns
    #             and not intersecting_geometries.empty
    #             else None
    #         )
    #
    #     for column in mean_columns:
    #         column = column.strip()
    #         chunk[f"{layer_name}::{column}::mean"] = chunk["intersecting"].apply(
    #             lambda intersecting_geometries: intersecting_geometries[column].mean()
    #         )
    #
    #     for column in sum_columns:
    #         column = column.strip()
    #         chunk[f"{layer_name}::{column}::sum"] = chunk["intersecting"].apply(
    #             lambda intersecting_geometries: intersecting_geometries[column].sum()
    #         )
    #
    #     for column in count_columns:
    #         column = column.strip()
    #         chunk[f"{layer_name}::{column}::count"] = chunk["intersecting"].apply(
    #             lambda intersecting_geometries: intersecting_geometries[column].count()
    #         )
    #
    #     return chunk
    #
    # chunk_size = 100
    # # Process grid_gdf in chunks
    # n_chunks = int(np.ceil(grid_gdf.shape[0] / chunk_size))
    # grid_gdf_chunks = np.array_split(grid_gdf, n_chunks)
    #
    # final_cells = []
    # for chunk in tqdm(grid_gdf_chunks, total=n_chunks):
    #     final_cells.append(process_chunk(chunk))
    #
    # grid_gdf_processed = pd.concat(final_cells)

    grid_gdf.crs = gdf.crs
    logger.info("Calculating statistics...")
    sindex = gdf.sindex
    for i, cell in tqdm(grid_gdf.iterrows(), total=grid_gdf.shape[0]):
        # Use spatial index to get possible matches and reduce the number of geometries to check
        possible_matches_index = list(sindex.intersection(cell.geometry.bounds))
        possible_matches = gdf.iloc[possible_matches_index]

        # Now apply the intersects function only to these geometries
        intersecting_geometries = possible_matches[
            possible_matches.intersects(cell.geometry)
        ].copy()

        # Calculate the intersection area
        intersecting_geometries["intersection_area"] = (
            intersecting_geometries.geometry.intersection(cell.geometry).area
        )

        for column in mode_columns:
            column = column.strip()
            mode = intersecting_geometries[column].mode()
            grid_gdf.loc[i, f"{layer_name}::{column}::mode"] = (
                mode.iloc[0] if not mode.empty else None
            )

        for column in most_intersecting_columns:
            column = column.strip()
            if column in intersecting_geometries:
                grouped = intersecting_geometries.groupby(column)[
                    "intersection_area"
                ].sum()
                if not grouped.empty:
                    max_value = grouped.idxmax()
                    grid_gdf.loc[i, f"{layer_name}::{column}::most_intersecting"] = (
                        max_value
                    )

        for column in mean_columns:
            column = column.strip()
            mean = intersecting_geometries[column].mean()
            grid_gdf.loc[i, f"{layer_name}::{column}::mean"] = mean

        for column in sum_columns:
            column = column.strip()
            total = intersecting_geometries[column].sum()
            grid_gdf.loc[i, f"{layer_name}::{column}::sum"] = total

        for column in count_columns:
            column = column.strip()
            count = intersecting_geometries[column].count()
            grid_gdf.loc[i, f"{layer_name}::{column}::count"] = count

    data_to_smooth = []
    for attribute in (
        mode_columns
        + most_intersecting_columns
        + mean_columns
        + sum_columns
        + count_columns
    ):
        grid_attribute = next(
            column
            for column in grid_gdf.columns
            if column.startswith(f"{layer_name}::{attribute}")
        )
        # fastest way to find out if it contains categorical data, even if int or float
        random_values = grid_gdf[grid_attribute].sample(1000)
        unique_values = random_values.unique()
        if len(unique_values) < 20:
            data_to_smooth.append(grid_attribute)

    for attribute in data_to_smooth:
        np_grid = grid_to_numpy(grid_gdf, attribute)
        if np_grid.dtype == "object":
            unique_map = {value: idx for idx, value in enumerate(np.unique(np_grid.astype(str)))}
            # map strings to ints to support ndimage
            np_grid = np.vectorize(unique_map.get)(np_grid.astype(str)).astype(np.int32)
            was_mapped = True
        else:
            was_mapped = False

        def apply_mode(values):
            return scipy.stats.mode(values)[0]

        smoothed_grid = ndimage.generic_filter(np_grid, apply_mode, size=3)

        if was_mapped:
            # map back to strings
            smoothed_grid = np.vectorize({v: k for k, v in unique_map.items()}.get)(
                smoothed_grid
            )

        grid_gdf = apply_np_to_grid(
            grid_gdf, smoothed_grid, attribute + "::mode_kernel3"
        )

    grid_gdf.to_file(
        input_gpkg, layer=cfg.layers.grid.name, driver="GPKG", engine="pyogrio"
    )
