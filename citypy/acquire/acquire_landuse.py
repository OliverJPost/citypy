import geopandas as gpd
import osmnx
import pandas as pd
from loguru import logger
from osmnx._errors import InsufficientResponseError
from shapely import Polygon

LANDUSE_CLASSES = {
    "residential": ["residential"],
    "industrial": ["industrial", "factory", "manufacture", "warehouse"],
    "retail": ["retail", "shop", "supermarket", "mall"],
    "commercial": ["commercial", "office", "bank", "government", "institutional"],
    # "green": [
    #     "nature",
    #     "park",
    #     "forest",
    #     "wood",
    #     "wetland",
    #     "meadow",
    #     "grass",
    #     "allotments",
    #     "farmland",
    #     "animal_keeping",
    #     "flowerbed",
    #     "farmyard",
    #     "vineyard",
    #     "orchard",
    #     "cemetery",
    #     "recreation_ground",
    #     "garden",
    #     "conservation",
    #     "recreation",
    #     "playground",
    #     "pitch",
    #     "sports",
    #     "golf_course",
    #     "meadow",
    # ],
}


def merge_landuse_classes(landuse: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    landuse["landuse_class"] = None
    for landuse_class, landuse_types in LANDUSE_CLASSES.items():
        landuse.loc[landuse["landuse"].isin(landuse_types), "landuse_class"] = (
            landuse_class
        )
    return landuse


ESSENTIAL_LANDUSE_COLUMNS = ["geometry", "landuse", "landuse_class"]


def filter_essential_columns(landuse_combined):
    return landuse_combined.filter(ESSENTIAL_LANDUSE_COLUMNS)


def get_landuse(extents: Polygon, local_crs, essentials_only=True) -> gpd.GeoDataFrame:
    logger.info("Downloading landuse data...")
    landuse_gdf = download_landuse(extents, local_crs)
    landuse_gdf = merge_landuse_classes(landuse_gdf)
    airports_gdf = download_airports(extents, landuse_gdf.crs)

    # Combine the GeoDataFrames
    combined_gdf = merge_landuse(airports_gdf, landuse_gdf)

    if essentials_only:
        combined_gdf = filter_essential_columns(combined_gdf)
    return combined_gdf


def merge_landuse(airports_gdf, landuse_gdf):
    airports_gdf["landuse"] = airports_gdf["landuse_class"] = "airport"
    combined_gdf = gpd.GeoDataFrame(
        pd.concat([landuse_gdf, airports_gdf], ignore_index=True)
    )
    combined_gdf = combined_gdf.to_crs(landuse_gdf.crs)
    return combined_gdf


def download_airports(extents, crs):
    try:
        airports_gdf = osmnx.features_from_polygon(extents, {"aeroway": "aerodrome"})
        airports_gdf = airports_gdf.to_crs(crs)
    except InsufficientResponseError:
        airports_gdf = gpd.GeoDataFrame()

    return airports_gdf


def download_landuse(extents, local_crs):
    landuse_gdf = osmnx.features_from_polygon(extents, {"landuse": True})
    landuse_gdf = landuse_gdf.to_crs(local_crs)
    return landuse_gdf
