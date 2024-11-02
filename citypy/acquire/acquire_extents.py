import geopandas as gpd
import osmnx
from shapely import Polygon

ESSENTIAL_EXTENTS_COLUMNS = ["geometry"]


def filter_essential_columns(extents):
    return extents.filter(ESSENTIAL_EXTENTS_COLUMNS)


def get_extents(
    city: str, country: str, essentials_only=True
) -> tuple[Polygon, gpd.GeoDataFrame]:
    extents_raw = osmnx.geocoder.geocode_to_gdf({"city": city, "country": country})
    extents = extents_raw.to_crs(extents_raw.estimate_utm_crs())

    if essentials_only:
        extents = filter_essential_columns(extents)

    return extents_raw["geometry"].unary_union, extents
