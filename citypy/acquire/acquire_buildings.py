import sys
from pathlib import Path
from typing import Optional

import geopandas as gpd
import numpy as np
import osmnx
import overturemaps
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
from loguru import logger
from overturemaps.core import geoarrow_schema_adapter
from pyarrow import fs
from shapely import LineString, Polygon

from citypy.acquire.acquire_ghsl import get_cropped_ghsl_data
from citypy.console import conditional_status
from citypy.util.raster import GeoRaster
from config import cfg


def is_rectangle(polygon):
    # Check if the polygon has 4 vertices
    if (
        len(polygon.exterior.coords) != 5
    ):  # 5 because the first and last point are the same
        return False

    # Extract the coordinates of the polygon
    coords = list(polygon.exterior.coords)[
        :-1
    ]  # Ignore the last point (same as the first)

    # Check if opposite sides are parallel and of equal length
    side1 = LineString([coords[0], coords[1]])
    side2 = LineString([coords[1], coords[2]])
    side3 = LineString([coords[2], coords[3]])
    side4 = LineString([coords[3], coords[0]])

    if not (side1.length == side3.length and side2.length == side4.length):
        return False

    # Check if all angles are 90 degrees
    def is_right_angle(a, b, c):
        ab = (b[0] - a[0], b[1] - a[1])
        bc = (c[0] - b[0], c[1] - b[1])
        dot_product = ab[0] * bc[0] + ab[1] * bc[1]
        return dot_product == 0

    return (
        is_right_angle(coords[0], coords[1], coords[2])
        and is_right_angle(coords[1], coords[2], coords[3])
        and is_right_angle(coords[2], coords[3], coords[0])
        and is_right_angle(coords[3], coords[0], coords[1])
    )


# From overturemaps-py, adapted for local storage
def record_batch_reader(
    bbox, theme="buildings", type="building"
) -> Optional[pa.RecordBatchReader]:
    """Return a pyarrow RecordBatchReader for the desired bounding box and s3 path."""
    xmin, ymin, xmax, ymax = bbox
    parquet_filter = (
        (pc.field("bbox", "xmin") < xmax)
        & (pc.field("bbox", "xmax") > xmin)
        & (pc.field("bbox", "ymin") < ymax)
        & (pc.field("bbox", "ymax") > ymin)
    )

    path = Path(cfg.local_overture.release_folder)
    path = path / f"theme={theme}" / f"type={type}"
    dataset = ds.dataset(path, filesystem=fs.LocalFileSystem())
    batches = dataset.to_batches(filter=parquet_filter)

    # to_batches() can yield many batches with no rows. I've seen
    # this cause downstream crashes or other negative effects. For
    # example, the ParquetWriter will emit an empty row group for
    # each one bloating the size of a parquet file. Just omit
    # them so the RecordBatchReader only has non-empty ones. Use
    # the generator syntax so the batches are streamed out
    non_empty_batches = (b for b in batches if b.num_rows > 0)

    geoarrow_schema = geoarrow_schema_adapter(dataset.schema)
    reader = pa.RecordBatchReader.from_batches(geoarrow_schema, non_empty_batches)
    return reader


def download_raw_overture_building_data(
    extents: Polygon, local_crs
) -> gpd.GeoDataFrame:
    bbox = extents.bounds

    if cfg.local_overture.use_local_parquet:
        reader = record_batch_reader(bbox)
        building_data = gpd.GeoDataFrame.from_arrow(reader)
    else:
        try:
            building_data = overturemaps.core.geodataframe("building", bbox)
        except OSError as e:
            logger.critical(e)
            logger.critical(
                "Could not download building data from Overture Maps. If this issue persists,"
                "turn off overturemaps.use_overture_for_buildings in settings.toml"
            )
            sys.exit(1)

    building_data.crs = "EPSG:4326"

    # if not building_data.sindex:
    #     raise Exception("No spatial index for overture data")

    if not is_rectangle(extents):
        logger.info("Extents are not rectangular, cropping...")
        building_data = building_data.clip(extents)
    building_data = building_data.to_crs(local_crs)
    return building_data


def download_raw_osm_building_data(extents: Polygon, local_crs) -> gpd.GeoDataFrame:
    buildings = osmnx.features_from_polygon(extents, tags={"building": True})
    buildings = buildings.to_crs(local_crs)
    return buildings


def only_keep_polygons(building_data: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    return building_data[building_data.geom_type == "Polygon"].reset_index(drop=True)


OSM_ATTRIBUTES_TO_KEEP = {
    "geometry": "geometry",
    "building": cfg.layers.buildings.base_attributes["class"],
    "height": cfg.layers.buildings.base_attributes.real_height,
    "building:levels": cfg.layers.buildings.base_attributes.real_levels,
    "roof:shape": cfg.layers.buildings.base_attributes.roof_shape,
    "name": cfg.layers.buildings.base_attributes.name,
}

OVERTURE_SUBTYPES_TO_OSM_BUILDING = {
    "agricultural": [
        "barn",
        "conservatory",
        "cowshed",
        "farm_auxiliary",
        "greenhouse",
        "slurry_tank",
        "stable",
        "sty",
        "livestock",
    ],
    "civic": [
        "bakehouse",
        "bridge",
        "civic",
        "fire_station",
        "government",
        "gatehouse",
        "kindergarten",
        "museum",
        "public",
        "toilets",
        "train_station",
        "transportation",
    ],
    "commercial": [
        "commercial",
        "kiosk",
        "office",
        "retail",
        "supermarket",
        "warehouse",
    ],
    "education": [
        "school",
        "university",
        "college",
    ],
    "entertainment": [
        "grandstand",
        "pavilion",
        "riding_hall",
        "sports_hall",
        "sports_centre",
        "stadium",
    ],
    "industrial": ["industrial"],
    "medical": [
        "hospital",
    ],
    "military": [],
    "outbuilding": ["allotment_house", "boathouse", "hangar", "hut", "shed"],
    "religious": [
        "religious",
        "cathedral",
        "chapel",
        "church",
        "kingdom_hall",
        "monastery",
        "mosque",
        "presbytery",
        "shrine",
        "synagogue",
        "temple",
    ],
    "residential": [
        "apartments",
        "barracks",
        "bungalow",
        "cabin",
        "detached",
        "annexe",
        "dormitory",
        "farm",
        "ger",
        "hotel",
        "house",
        "houseboat",
        "residential",
        "semidetached_house",
        "static_caravan",
        "stilt_house",
        "terrace",
        "tree_house",
        "trullo",
    ],
    "service": [],
    "transportation": [],
}


def reverse_dict(d):
    reversed_dict = {}
    for key, value_list in d.items():
        for value in value_list:
            reversed_dict[value] = key
    return reversed_dict


OSM_BUILDING_TAG_TO_OVERTURE_SUBTYPE = reverse_dict(OVERTURE_SUBTYPES_TO_OSM_BUILDING)


def clean_osm_attributes(footprints: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    return footprints.filter(OSM_ATTRIBUTES_TO_KEEP.keys())


def osm_to_overture_schema(footprints: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    data = footprints.rename(columns=OSM_ATTRIBUTES_TO_KEEP, errors="ignore")
    data[cfg.layers.buildings.base_attributes.subtype] = data[
        cfg.layers.buildings.base_attributes["class"]
    ].apply(lambda cls: OSM_BUILDING_TAG_TO_OVERTURE_SUBTYPE.get(cls))
    return data


def get_osm_building_data(extents: Polygon, local_crs) -> gpd.GeoDataFrame:
    raw_data = download_raw_osm_building_data(extents, local_crs)
    footprints = only_keep_polygons(raw_data)
    footprints_cleaned = clean_osm_attributes(footprints)
    footprints_namespaced = osm_to_overture_schema(footprints_cleaned)
    return footprints_namespaced


def add_overture_source_column(data: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    def map_source(source_list):
        if source_list is None or len(source_list) == 0:
            return None
        return source_list[0].get("dataset")

    def map_confidence(source_list):
        if source_list is None or len(source_list) == 0:
            return None
        return source_list[0].get("confidence")

    data["source"] = data["sources"].apply(map_source)
    data["confidence"] = data["sources"].apply(map_confidence)

    return data


def get_overture_building_data(extents: Polygon, local_crs) -> gpd.GeoDataFrame:
    raw_data = download_raw_overture_building_data(extents, local_crs)
    data = add_overture_source_column(raw_data)
    data = merge_overture_attributes(data)
    return data


def merge_overture_attributes(data: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    data[cfg.layers.buildings.base_attributes.real_height] = data["height"].where(
        data["height"].notna(), data.get("roof_height")
    )
    data[cfg.layers.buildings.base_attributes.real_levels] = data.get("num_floors")
    data[cfg.layers.buildings.base_attributes.roof_shape] = data.get("roof_shape")
    data[cfg.layers.buildings.base_attributes["class"]] = data.get("class")
    data[cfg.layers.buildings.base_attributes.subtype] = data.get("subtype")
    data[cfg.layers.buildings.base_attributes.name] = (
        data["names"].get("primary") if data.get("names").empty else None
    )

    return data


def combine_attributes(joined_buildings: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    # FIXME joined_buildings = merge_level_attribute(joined_buildings)
    return joined_buildings


def add_unique_id(buildings: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    # create column called "id::building" from index
    buildings[cfg.layers.buildings.unique_id] = buildings.index
    # buildings.set_index(cfg.layers.buildings.unique_id, inplace=True)
    return buildings


ESSENTIAL_BUILDING_COLUMNS = [
    "geometry",
    cfg.layers.buildings.unique_id,
    *cfg.layers.buildings["base_attributes"].values(),
]


def filter_essential_columns(buildings):
    return buildings.filter(ESSENTIAL_BUILDING_COLUMNS)


def add_ghsl_height(
    buildings: gpd.GeoDataFrame, ghsl_raster: GeoRaster
) -> gpd.GeoDataFrame:
    ghsl_raster = ghsl_raster.reprojected(buildings.crs)
    buildings[cfg.layers.buildings.base_attributes.ghsl_height] = (
        buildings.geometry.apply(
            lambda g: ghsl_raster.sample(g.centroid.x, g.centroid.y)[0]
        )
    )

    return buildings


def get_building_data(
    extents: Polygon, local_crs, essentials_only=True, verbose=True
) -> gpd.GeoDataFrame:
    # Possibility to add other providers
    with conditional_status(verbose) as status:
        status.update("Downloading building footprints...")
        logger.info("Downloading building footprints...")
        if cfg.overturemaps.use_overture_for_buildings:
            joined_buildings = get_overture_building_data(extents, local_crs)
        else:
            joined_buildings = get_osm_building_data(extents, local_crs)
        joined_buildings = combine_attributes(joined_buildings)
        joined_buildings = add_unique_id(joined_buildings)
        if essentials_only:
            joined_buildings = filter_essential_columns(joined_buildings)
    logger.info("Downloading building height data...")
    extents_df = gpd.GeoDataFrame(geometry=[extents], crs="EPSG:4326")
    logger.info("Merging building height data...")
    ghsl_h = get_cropped_ghsl_data(extents_df, "BUILT-H ANBH")
    joined_buildings = add_ghsl_height(joined_buildings, ghsl_h)
    return joined_buildings
