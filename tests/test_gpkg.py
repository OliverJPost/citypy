import uuid

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import rasterio
from shapely import Point

from citypy.acquire.acquire_ghsl import get_cropped_ghsl_data
from citypy.util.gpkg import GeoPackage
from citypy.util.raster import reproject_raster


@pytest.mark.skip(reason="Functionality not working, test doesn't catch it")
def test_drop_gpkg_table_if_exists_vector(tmp_path):
    gdf1 = gpd.GeoDataFrame(
        {"geometry": [Point(0, 0), Point(1, 1)]},
        crs="EPSG:4326",
    )
    gdf2 = gpd.GeoDataFrame(
        {"geometry": [Point(0, 0), Point(1, 1)]},
        crs="EPSG:4326",
    )
    gdf1.to_file(str(tmp_path / "test.gpkg"), layer="test", driver="GPKG")
    gdf2.to_file(str(tmp_path / "test.gpkg"), layer="test2", driver="GPKG")

    # Test that the layer exists
    gpd.read_file(tmp_path / "test.gpkg", layer="test2", driver="GPKG")

    gpkg = GeoPackage(tmp_path / "test.gpkg")
    gpkg.drop_table("test2")

    with pytest.raises(ValueError):
        gpd.read_file(tmp_path / "test.gpkg", layer="test2", driver="GPKG")


@pytest.mark.skip(reason="Functionality not working, test doesn't catch it")
def test_drop_gpkg_table_if_exists_raster(tmp_geopackage_path, georaster):
    gpkg = GeoPackage(tmp_geopackage_path)
    gpkg.write_raster(georaster, "test_raster")
    assert gpkg.read_raster("test_raster").raster is not None
    gpkg.drop_table("test_raster")
    with pytest.raises(rasterio.errors.RasterioIOError):
        gpkg.read_raster("test_raster")


def test_read_raster_from_gpkg(tmp_geopackage_path, georaster):
    gpkg = GeoPackage(tmp_geopackage_path)
    gpkg.write_raster(georaster, "test_raster")
    raster = gpkg.read_raster("test_raster")
    assert np.array_equal(raster.raster, georaster.raster)
    assert raster.crs == georaster.crs
    assert raster.transform == georaster.transform
    assert raster.raster.dtype == georaster.raster.dtype


def test_read_vector_from_gpkg(tmp_geopackage_path):
    gdf = gpd.GeoDataFrame(
        {"geometry": [Point(0, 0), Point(1, 1)]},
        crs="EPSG:4326",
    )
    gdf.to_file(tmp_geopackage_path, layer="test", driver="GPKG")

    gpkg = GeoPackage(tmp_geopackage_path)
    gdf_read = gpkg.read_vector("test")
    assert gdf_read.crs == gdf.crs
    assert gdf_read.equals(gdf)


def test_write_vector_to_gpkg(tmp_geopackage_path):
    gdf = gpd.GeoDataFrame(
        {"geometry": [Point(0, 0), Point(1, 1)]},
        crs="EPSG:4326",
    )

    gpkg = GeoPackage(tmp_geopackage_path)
    gpkg.write_vector(gdf, "test")
    gdf_read = gpkg.read_vector("test")
    assert gdf_read.crs == gdf.crs
    assert gdf_read.equals(gdf)


# def test_write_attribute_table_to_gpkg_existing(tmp_geopackage_path):
#     df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
#     gpkg = GeoPackage(tmp_geopackage_path)
#     layer_name = "test_layer"
#     gpkg.write_attribute_table(df, layer_name)
#     df_read = gpkg.read_attribute_table(layer_name)
#     print(df_read)
#     pd.testing.assert_frame_equal(df, df_read)


def test_list_raster_layers(tmp_citydata_gpkg_path, georaster):
    layer_name_uuid = str(uuid.uuid4())
    gpkg = GeoPackage(tmp_citydata_gpkg_path)
    gpkg.write_raster(georaster, layer_name_uuid)
    layers = gpkg.list_raster_layers()
    assert layer_name_uuid in layers


@pytest.mark.skip(reason="Test not finished")
def test_write_raster_to_gpkg(extents):
    tile, transform = get_cropped_ghsl_data(extents, "BUILT-H ANBH")
    reprojected_tile, reprojected_transform = reproject_raster(
        tile,
        transform,
        rasterio.crs.CRS.from_epsg(4326),  # type: ignore
        extents.crs,
    )

    # write_raster_to_gpkg(
    #     reprojected_tile,
    #     extents.crs,
    #     reprojected_transform,
    #     Path(
    #         "/Users/ole/Library/Mobile Documents/com~apple~CloudDocs/Areas/Geomatics/Thesis/2023-Post/citypy/pijnacker_nl_data.gpkg"
    #     ),
    #     "test_raster16",
    # )


# @pytest.mark.skip(reason="Test not finished")
# def test_write_raster_to_gpkg_multiband(extents):
#     tile, transform = get_cropped_ghsl_data(extents, "BUILT-H ANBH")
#     reprojected_tile, reprojected_transform = reproject_raster(
#         tile, transform, rasterio.crs.CRS.from_epsg(4326), extents.crs
#     )
#     write_raster_to_gpkg(
#         reprojected_tile,
#         extents.crs,
#         reprojected_transform,
#         Path(
#             "/Users/ole/Library/Mobile Documents/com~apple~CloudDocs/Areas/Geomatics/Thesis/2023-Post/citypy/pijnacker_nl_data.gpkg"
#         ),
#         "test_raster16",
#     )
