import sys

import geopandas as gpd
import overturemaps
from loguru import logger

from config import cfg

from .acquire_buildings import record_batch_reader
from .acquire_raster import get_raster


def get_land_cover(extents, local_crs):
    logger.info("Downloading land cover data...")
    bbox = extents.to_crs("EPSG:3857")
    bbox = ",".join(bbox.bounds)
    # 25m pixel resolution
    extents_meters = extents.to_crs(local_crs)
    width_m = extents_meters.bounds[2] - extents_meters.bounds[0]
    height_m = extents_meters.bounds[3] - extents_meters.bounds[1]
    width_px = width_m / 25
    height_px = height_m / 25

    url = (
        f"https://services.terrascope.be/wms/v2?"
        f"service=WMS&request=GetMap&version=1.1.1&BGCOLOR=0xFFFFFF"
        f"&crs=EPSG:3857"
        f"&srs=EPSG:3857"
        f"&bbox={bbox}"
        f"&layers=WORLDCOVER_2021_MAP"
        f"&width={width_px}&height={height_px}"
        f"&format=image/png"
    )

    img = get_raster(url)

    return df.to_crs(local_crs)
