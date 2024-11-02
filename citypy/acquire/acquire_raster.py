import io
import uuid
import zipfile
from pathlib import Path

import geopandas as gpd
import rasterio
import rasterio.mask
import requests
from rasterio import CRS
from shapely import Polygon

from citypy.util.raster import GeoRaster
from config import cfg

# TODO config
CACHE_DIR = Path(__file__).parent / "cache"


def get_terrain_raster(extents: Polygon) -> GeoRaster:
    west, south, east, north = extents.bounds
    api_key = cfg.opentopography.api_key
    url = f"https://portal.opentopography.org/API/globaldem?demtype=COP90&south={south}&north={north}&west={west}&east={east}&outputFormat=GTiff&API_Key={api_key}"
    raster = get_raster(url)
    return raster


def get_raster(url: str) -> GeoRaster:
    if not exists_in_cache(url):
        download_raster(url)

    return GeoRaster.from_file(path_in_cache(url))


def get_raster_cropped(url: str, extents: gpd.GeoDataFrame) -> GeoRaster:
    if not exists_in_cache(url):
        download_raster(url)

    raster = rasterio.open(path_in_cache(url))
    cropped_raster, cropped_transform = rasterio.mask.mask(
        raster, extents.geometry, crop=True
    )
    return GeoRaster(cropped_raster, cropped_transform, raster.crs)


def download_raster(url: str) -> None:
    response = requests.get(url)
    response.raise_for_status()
    if not CACHE_DIR.exists():
        CACHE_DIR.mkdir()

    # Check if response is zip or tif
    if response.headers["Content-Type"] == "application/zip":
        unzipped = zipfile.ZipFile(io.BytesIO(response.content))
        tif_files = [f for f in unzipped.namelist() if f.endswith(".tif")]
        if len(tif_files) != 1:
            raise ValueError(f"Expected 1 .tif file, got {len(tif_files)}")
        image = unzipped.read(tif_files[0])
    elif response.headers["Content-Type"] in ("image/tiff", "application/octet-stream"):
        image = response.content
    elif response.headers["Content-Type"] == "image/png;charset=UTF-8":
        image_png = response.content
        image_png = rasterio.io.MemoryFile(image_png, ext="png").open().read()
        image = io.BytesIO()
        with rasterio.open(image) as src:
            rasterio.shutil.copy(src, image_png, driver="GTiff")
    else:
        raise ValueError(
            f"Expected response to be either a zip file or a tif file, got {response.headers['Content-Type']}"
        )

    filename = str(uuid_from_url(url)) + ".tif"
    with open(CACHE_DIR / filename, "wb") as f:
        f.write(image)

    # TODO max size for cache


def uuid_from_url(url):
    return uuid.uuid5(uuid.NAMESPACE_URL, url)


def exists_in_cache(url):
    return path_in_cache(url).exists()


def path_in_cache(url):
    return CACHE_DIR / (str(uuid_from_url(url)) + ".tif")
