from pathlib import Path
from typing import Literal

import geopandas as gpd
import numpy as np

from citypy.acquire.acquire_raster import get_raster, get_raster_cropped
from citypy.console import conditional_status
from citypy.util.raster import GeoRaster

TILE_DIR = Path(__file__).parent / "data_tiling"

DATASET_NAMES = Literal["BUILT-H AGBH", "BUILT-H ANBH"]


def get_intersecting_ghsl_wgs84_tiles(extents: gpd.GeoDataFrame) -> list[str]:
    tile_index = gpd.read_file(TILE_DIR / "GHSL" / "WGS84_tile_schema.shp")
    extents = extents.to_crs(tile_index.crs)

    intersecting_tiles = gpd.sjoin(
        left_df=extents,
        right_df=tile_index,
        how="left",
        predicate="intersects",
    )
    intersecting_tile_ids = intersecting_tiles["tile_id"].unique()
    return intersecting_tile_ids


def get_cropped_ghsl_data(
    extents: gpd.GeoDataFrame, dataset: DATASET_NAMES, verbose: bool = True
) -> GeoRaster:
    with conditional_status(
        verbose, message=f"Downloading GHSL dataset: {dataset}"
    ) as status:
        intersecting_tiles = get_intersecting_ghsl_wgs84_tiles(extents)
        tile_rasters = []
        for tile_id in intersecting_tiles:
            cropped_tile = _get_cropped_ghsl_tile(tile_id, dataset, extents)
            reprojected_tile = cropped_tile.reprojected(extents.crs)
            tile_rasters.append(reprojected_tile)

        if len(tile_rasters) > 1:
            raster = GeoRaster.from_tiles(tile_rasters)
        else:
            raster = tile_rasters[0]

    return raster


def _get_cropped_ghsl_tile(
    tile_id: str, dataset: DATASET_NAMES, extents: gpd.GeoDataFrame
) -> GeoRaster:
    url = get_ghsl_url(dataset, tile_id)
    raster = get_raster_cropped(url, extents)
    return raster


def get_ghsl_url(dataset, tile_id):
    base_url = "https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL"
    if dataset.startswith("BUILT-H"):
        mean_type = dataset.split(" ")[-1]
        if mean_type not in ["AGBH", "ANBH"]:
            raise ValueError("mean_type must be either AGBH or ANBH")
        dataset_path = f"GHS_BUILT_H_GLOBE_R2023A/GHS_BUILT_H_{mean_type}_E2018_GLOBE_R2023A_4326_3ss/V1-0"
        tile_name = (
            f"GHS_BUILT_H_{mean_type}_E2018_GLOBE_R2023A_4326_3ss_V1_0_{tile_id}"
        )
    url = f"{base_url}/{dataset_path}/tiles/{tile_name}.zip"
    return url
