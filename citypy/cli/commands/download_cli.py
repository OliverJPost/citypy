import logging
import re
import sys
from enum import Enum
from pathlib import Path
from typing import List

import geopandas as gpd
import osmnx
import typer
from geopy import Nominatim
from shapely import Polygon

from citypy.acquire import get_building_data, get_road_data
from citypy.acquire.acquire_city_center import get_city_center
from citypy.acquire.acquire_extents import get_extents
from citypy.acquire.acquire_ghsl import get_cropped_ghsl_data
from citypy.acquire.acquire_land_cover import get_land_cover
from citypy.acquire.acquire_landuse import get_landuse
from citypy.acquire.acquire_raster import get_terrain_raster
from citypy.acquire.acquire_water import get_water_data
from citypy.cli import app
from citypy.describe import display_gpkg_description
from citypy.logging_setup import logger
from citypy.process import process_landuse
from citypy.util.gpkg import (
    GeoPackage,
    LockReasons,
)
from config import cfg

geolocator = Nominatim(user_agent="citypy", timeout=20)


def sanetize(filename: str) -> str:
    return re.sub(r"[^\w-]", "_", filename)


class DownloadError(Enum):
    AREA_TO_BIG = 3
    COULD_NOT_BE_GEOCODED = 4
    GEOCODED_POLYGON_NOT_FOUND = 5
    INVALID_BBOX = 6
    GEOCODE_NOT_IN_BBOX = 7

    def __str__(self):
        return str(self.value)


@app.command()
def download(
    city: str = typer.Argument(
        help="Name of the city you want to acquire the data from.",
    ),
    country: str = typer.Argument(help="Name or ISO code of country this city is in."),
    essentials_only: bool = typer.Option(
        True,
        "--essentials-only/--all-columns",
        help="Only keep the columns deemed essential. Essentials defined in config file.",
    ),
    bbox: str = typer.Option(
        None,
        help="CSV formatted west,south,east,north bounding box in WGS84. String without spaces.",
    ),
    overwrite: bool = typer.Option(
        True,
        "--overwrite/--no-overwrite",
        help="Overwrite existing GeoPackage.",
    ),
    buildings: bool = typer.Option(
        True,
        help="Download building data.",
    ),
    roads: bool = typer.Option(
        True,
        help="Download road data.",
    ),
    landuse: bool = typer.Option(
        True,
        help="Download landuse data.",
    ),
    terrain: bool = typer.Option(
        True,
        help="Download GHSL data.",
    ),
    city_center: bool = typer.Option(
        True,
        help="Download city center data.",
    ),
    water: bool = typer.Option(
        True,
        help="Download water data.",
    ),
    extents_layer: bool = typer.Option(
        True,
        help="Download extents data.",
    ),
    land_cover: bool = typer.Option(
        True,
        help="Download land cover data.",
    ),
) -> None:
    if not hasattr(cfg, "opentopography"):
        raise Exception("Secrets file missing for OpenTopography API key.")

    city = city.strip().capitalize()
    country = country.strip().upper()
    output_file = Path(sanetize(f"{city}_{country}") + ".gpkg")
    location = geolocator.geocode({"city": city, "country": country})
    if location is None:
        logger.critical(f"'{city}', '{country}' couldn't be geocoded.")
        sys.exit(DownloadError.COULD_NOT_BE_GEOCODED)

    if bbox:
        # Validate string
        if not re.match(
            r"^-?\d+(\.\d+)?,-?\d+(\.\d+)?,-?\d+(\.\d+)?,-?\d+(\.\d+)?$", bbox
        ):
            logger.critical(f"Invalid bbox format: '{bbox}'")
            sys.exit(DownloadError.INVALID_BBOX)
        west, south, east, north = map(float, bbox.split(","))
        if not (west < location.longitude < east and south < location.latitude < north):
            logger.critical(
                f"Location of '{city}', '{country}' is not within the bbox."
            )
            sys.exit(DownloadError.GEOCODE_NOT_IN_BBOX)
        extends_polygon = Polygon.from_bounds(west, south, east, north)
        extents_gdf = gpd.GeoDataFrame(geometry=[extends_polygon], crs="EPSG:4326")
    else:
        try:
            extents, extents_gdf = get_extents(
                city, country, essentials_only=essentials_only
            )
        except osmnx._errors.InsufficientResponseError:
            logger.critical(f"'{city}', '{country}' couldn't be geocoded.")
            sys.exit(DownloadError.COULD_NOT_BE_GEOCODED)
        except TypeError as e:
            if str(e).endswith("to a geometry of type (Multi)Polygon"):
                logger.critical(
                    f"'{city}', '{country}' couldn't be geocoded to a polygon."
                )
                exit(DownloadError.GEOCODED_POLYGON_NOT_FOUND)
            raise e

    # extents as bbox of extents polygon
    bounds = extents_gdf.to_crs("EPSG:4326").total_bounds

    # reject if area is bigger than 25 kilometers in any direction. it's in degrees wg84!!
    if max(bounds[2] - bounds[0], bounds[3] - bounds[1]) > 0.225:
        logger.warning("Area is bigger than 25x25 km.")
        # exit(DownloadError.AREA_TO_BIG)

    extents = Polygon.from_bounds(*bounds)
    local_crs = extents_gdf.estimate_utm_crs()

    gpkg = GeoPackage(output_file, verbose=True)
    if gpkg.is_locked() and not overwrite:
        # Existing GeoPackages should be deleted.
        logger.warning(f"{gpkg} already exists. Deleting...")
        gpkg.delete_existing_file()
    # Lock geopackage to detect interrupted downloads
    gpkg.lock(reason=LockReasons.DOWNLOAD)

    if extents_layer:
        gpkg.write_vector(extents_gdf, cfg.layers.extents.name)

    if buildings:
        building_data = get_building_data(
            extents, local_crs, essentials_only=essentials_only, verbose=True
        )
        logger.info(f"Writing {len(building_data)} buildings to GeoPackage.")
        gpkg.write_vector(building_data, cfg.layers.buildings.name)
    if roads:
        nodes, edges = get_road_data(
            extents, local_crs, essentials_only=essentials_only, verbose=True
        )
        logger.info(f"Writing {len(nodes)} nodes and {len(edges)} edges to GeoPackage.")
        gpkg.write_vector(nodes, cfg.layers.road_nodes.name)
        gpkg.write_vector(edges, cfg.layers.road_edges.name)
    if landuse:
        landuse = get_landuse(extents, local_crs, essentials_only=essentials_only)
        logger.info(f"Writing {len(landuse)} landuse polygons to GeoPackage.")
        gpkg.write_vector(landuse, cfg.layers.landuse.name)

    if city_center:
        city_center_df = get_city_center(location, city, local_crs)
        gpkg.write_vector(city_center_df, cfg.layers.city_center.name)

    if water:
        water_df = get_water_data(extents, local_crs, essentials_only=essentials_only)
        logger.info(f"Writing {len(water_df)} water polygons to GeoPackage.")
        gpkg.write_vector(water_df, cfg.layers.water.name)

    # if land_cover:
    #     land_cover_df = get_land_cover(extents, local_crs)
    #     gpkg.write_vector(land_cover_df, cfg.layers.land_cover.name)

    if terrain:
        terrain = get_terrain_raster(extents)
        terrain = terrain.reprojected(local_crs)
        terrain_gdf = terrain.to_gdf()
        logger.info("Writing terrain raster to GeoPackage.")
        gpkg.write_vector(terrain_gdf, "terrain")

    gpkg.unlock()
    display_gpkg_description(output_file)
