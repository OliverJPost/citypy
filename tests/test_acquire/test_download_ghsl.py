import geopandas as gpd
import pytest

from citypy.acquire.acquire_ghsl import (
    get_intersecting_ghsl_wgs84_tiles,
)
from citypy.acquire.acquire_raster import download_raster, get_raster


def test_get_intersecting_ghsl_tiles(extents):
    tiles = get_intersecting_ghsl_wgs84_tiles(extents)
    assert len(tiles) == 1
    assert tiles[0] == "R4_C19"


@pytest.mark.needs_internet
def test_get_intersecting_ghsl_tiles_multiple_tiles():
    extents_metz = gpd.read_file("test_data/metz_fr_extents.gpkg", driver="GPKG")
    tiles = get_intersecting_ghsl_wgs84_tiles(extents_metz)
    assert len(tiles) == 2
    assert "R4_C19" in tiles
    assert "R5_C19" in tiles


def test_download_ghsl_data():
    download_raster("R4_C19", "BUILT-H")


def test_download_cop90():
    raster = get_raster(
        "https://portal.opentopography.org/API/globaldem?demtype=COP90&south=50&north=50.1&west=14.35&east=14.6&outputFormat=GTiff&API_Key="
    )
    print("hello")
    raster


def test_download_worldcover():
    raster = get_raster(
        "https://services.terrascope.be/wms/v2?service=WMS&request=GetMap&version=1.1.1&BGCOLOR=0xFFFFFF&crs=EPSG:3857&srs=EPSG:3857&bbox=615385.959,5762081.614,619338.349,5766605.281&layers=WORLDCOVER_2021_MAP&width=936&height=1071&format=image/png"
    )
    raster
