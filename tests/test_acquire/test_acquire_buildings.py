from pathlib import Path

import shapely

from citypy.acquire.acquire_buildings import (
    # BUILDING_ATTRIBUTES,
    OSM_ATTRIBUTES_TO_KEEP,
    add_unique_id,
    clean_osm_attributes,
    combine_attributes,
    download_raw_osm_building_data,
    download_raw_overture_building_data,
    filter_essential_columns,
    get_building_data,
    get_osm_building_data,
    only_keep_polygons,
    osm_to_overture_schema,
)
from citypy.util.gpkg import (
    GeoPackage,
)
from config import cfg

from .fixtures import mock_raw_building_data


def test_raw_overture_download(extents):
    data = download_raw_overture_building_data(extents)
    data.to_file("test_overture.gpkg")


def test_raw_osm_download(extents, raw_building_data, mock_raw_building_data):
    data = download_raw_osm_building_data(extents)
    data.to_file("test_osm.gpkg")
    assert data.equals(raw_building_data)


# FIXME test data only contains polygons
def test_only_keep_polygons(raw_building_data):
    count_before = len(raw_building_data)
    cleaned = only_keep_polygons(raw_building_data)
    assert count_before != len(raw_building_data)

    for _, feature in cleaned.iterrows():
        assert isinstance(feature["geometry"], shapely.Polygon)


def test_clean_osm_attributes(raw_building_data):
    cleaned = clean_osm_attributes(raw_building_data)
    assert set(cleaned.columns) == set(OSM_ATTRIBUTES_TO_KEEP.keys())


# def test_combine_attributes(raw_building_data):
#     cleaned = clean_osm_attributes(raw_building_data)
#     combined = combine_attributes(cleaned)
#     assert set(combined.columns) == set(BUILDING_ATTRIBUTES)


def test_namespace_osm_attributes(raw_building_data):
    cleaned = clean_osm_attributes(raw_building_data)
    namespaced = osm_to_overture_schema(cleaned)
    assert set(namespaced.columns) == set(OSM_ATTRIBUTES_TO_KEEP.values())


def test_get_osm_building_data(city_name, mock_raw_building_data):
    result = get_osm_building_data(city_name)
    assert result.crs == result.estimate_utm_crs()


def test_filter_essential_columns(raw_building_data):
    assert len(filter_essential_columns(raw_building_data).columns) < len(
        raw_building_data.columns
    )


def test_add_unique_id(raw_building_data):
    with_ids = add_unique_id(raw_building_data)
    assert cfg.layers.buildings.unique_id in with_ids.columns
    assert with_ids[cfg.layers.buildings.unique_id].dtype == "int64"
    # No duplicates
    assert len(with_ids[cfg.layers.buildings.unique_id].unique()) == len(with_ids)
    # is NOT index (because momepy needs access to the column)
    assert with_ids.index.name != cfg.layers.buildings.unique_id


def test_import_export(raw_building_data, tmp_path):
    fp = tmp_path / "buildings_test_imp_exp.gpkg"
    gpkg = GeoPackage(fp)
    gpkg.write_vector(raw_building_data, cfg.layers.buildings.name)
    imported = gpkg.read_vector(cfg.layers.buildings.name)

    assert imported.equals(raw_building_data)


def test_get_building_data(city_name, raw_building_data, mock_raw_building_data):
    result = get_building_data(city_name, essentials_only=False)
    assert set(result.columns) != set(raw_building_data.columns)
    assert result.crs == result.estimate_utm_crs()

    for _, feature in result.iterrows():
        assert isinstance(feature["geometry"], shapely.Polygon)
