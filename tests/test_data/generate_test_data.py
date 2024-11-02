from pathlib import Path

from citypy.acquire.acquire_buildings import download_raw_osm_building_data
from citypy.util.gpkg import GeoPackage

CITY_NAME = "Delfgauw, NL"
DIR = Path(".")


def generate_raw_building_data(city_name):
    gdf = download_raw_osm_building_data(city_name)
    gdf.to_file(str(DIR / "raw_buildings.gpkg"), driver="GPKG")


if __name__ == "__main__":
    generate_raw_building_data(CITY_NAME)
