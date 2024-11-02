import geopandas as gpd
import osmnx
from loguru import logger
from shapely import Polygon

ESSENTIAL_WATER_COLUMNS = ["geometry", "natural", "water"]  # , "water_type"


def filter_essential_columns(water_gdf):
    water_gdf = water_gdf.filter(ESSENTIAL_WATER_COLUMNS)
    return water_gdf


def get_water_data(
    extents: Polygon, local_crs, essentials_only=True
) -> gpd.GeoDataFrame:
    """Get water data for a city.

    Args:
        city_name (str): Name of the city.

    Returns:
        gpd.GeoDataFrame: GeoDataFrame containing water data.
    """
    logger.info("Downloading water data...")
    water_gdf = osmnx.features_from_polygon(
        extents,
        {"natural": ["water", "coastline"]},
    )
    water_gdf = water_gdf.to_crs(local_crs)

    if essentials_only:
        water_gdf = filter_essential_columns(water_gdf)

    return water_gdf
