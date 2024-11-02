import uuid
from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
import rasterio
from shapely import Point, Polygon

from citypy.cli.commands.process_cli import LazyHighSpatialWeights, LazySpatialWeights
from citypy.util.gpkg import GeoPackage
from citypy.util.raster import GeoRaster
from config import cfg


@pytest.fixture()
def city_name():
    yield "Delfgauw, NL"


@pytest.fixture()
def extents():
    # Punta Arenas, with missing building data in OSM
    bounds = [-70.9808, -53.1988, -70.8166, -53.1139]
    extents = Polygon.from_bounds(*bounds)
    yield extents


RASTER_BAND_DIMENSIONS = {
    "1 band": 1,
    "3 bands": 3,
    "5 bands": 5,
}
RASTER_ASPECT_RATIO = {
    "square": (10, 10),
    "landscape": (20, 10),
}

RASTER_DATA_TYPE = {
    "uint16": np.uint16,
    "int16": np.int16,
    "float32": np.float32,
}

combined_params = [
    (band_desc, aspect_ratio_desc, dtype_desc)
    for band_desc in RASTER_BAND_DIMENSIONS.keys()
    for aspect_ratio_desc in RASTER_ASPECT_RATIO.keys()
    for dtype_desc in RASTER_DATA_TYPE.keys()
]
ids = [
    f"{band_desc}_{aspect_ratio_desc}_{dtype_desc}"
    for band_desc, aspect_ratio_desc, dtype_desc in combined_params
]


@pytest.fixture(params=combined_params, ids=ids)
def georaster(request):
    band_desc, aspect_ratio_desc, dtype_desc = request.param
    bands = RASTER_BAND_DIMENSIONS[band_desc]
    width, height = RASTER_ASPECT_RATIO[aspect_ratio_desc]
    dtype = RASTER_DATA_TYPE[dtype_desc]

    raster = np.zeros((bands, height, width), dtype=dtype)
    transform = rasterio.transform.Affine(1, 0, 0, 0, -1, 0)
    crs = rasterio.crs.CRS.from_epsg(4326)
    return GeoRaster(raster, transform, crs)


@pytest.fixture()
def tmp_geopackage_path(tmp_path):
    gdf = gpd.GeoDataFrame(
        {"geometry": [Point(0, 0), Point(1, 1)]},
        crs="EPSG:4326",
    )
    random_id = str(uuid.uuid4())
    filepath = tmp_path / f"{random_id}.gpkg"
    gdf.to_file(filepath, layer=random_id, driver="GPKG")
    yield filepath


@pytest.fixture()
def tmp_citydata_gpkg_path(tmp_path):
    random_id = str(uuid.uuid4())
    filepath = tmp_path / f"{random_id}.gpkg"
    buildings = gpd.GeoDataFrame(
        {"geometry": [Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])]},
        crs="EPSG:4326",
    )
    road_nodes = gpd.GeoDataFrame(
        {"geometry": [Point(0, 0), Point(1, 1)], "osmid": [0, 1]},
        crs="EPSG:4326",
    )
    road_edges = gpd.GeoDataFrame(
        {
            "geometry": [Point(0, 0), Point(1, 1)],
            "u": [0, 1],
            "v": [1, 0],
            "key": [0, 0],
        },
        crs="EPSG:4326",
    )
    gpkg = GeoPackage(filepath)
    gpkg.write_vector(buildings, cfg.layers.buildings.name)
    gpkg.write_vector(road_nodes, cfg.layers.road_nodes.name)
    gpkg.write_vector(road_edges, cfg.layers.road_edges.name)
    return filepath


@pytest.fixture()
def raw_building_data():
    yield gpd.read_file("test_data/raw_buildings.gpkg", engine="pyogrio")


@pytest.fixture
def processed_city():
    raise NotImplementedError("Not implemented yet.")
    # yield CityData.from_gpkg("test_data/delfgauw_nl_data_PROCESS.gpkg")


@pytest.fixture
def queen1(processed_city):
    yield LazySpatialWeights(processed_city.tesselation)


@pytest.fixture
def queen3(queen1):
    yield LazyHighSpatialWeights(queen1)
