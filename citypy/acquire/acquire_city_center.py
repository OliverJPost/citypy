import geopandas as gpd
from geopy.geocoders import Nominatim
from loguru import logger
from shapely.geometry import Point

geolocator = Nominatim(user_agent="randomcity")


def get_city_center(location, city, local_crs) -> gpd.GeoDataFrame:
    point = Point(location.longitude, location.latitude)
    gdf = gpd.GeoDataFrame({"name": [city], "geometry": [point]}, crs="EPSG:4326")
    return gdf.to_crs(local_crs)
