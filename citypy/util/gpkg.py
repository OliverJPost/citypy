import logging
import sqlite3
from contextlib import contextmanager
from enum import Enum, StrEnum
from os import PathLike
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pyogrio
import rasterio
from loguru import logger
from osgeo import ogr

from citypy.console import conditional_status, console, display_status_if_verbose
from citypy.util.raster import GeoRaster


class LockReasons(StrEnum):
    DOWNLOAD = "Download phase not completed. Either still processing or crashed."
    PROCESS = "Process phase not completed. Either still processing or crashed."
    CONTEXTUALIZE = (
        "Contextualize phase not completed. Either still processing or crashed."
    )


class GeoPackage:
    def __init__(self, filepath: PathLike, verbose=True):
        self.filepath = Path(filepath)
        self.verbose = verbose
        self._engine = "pyogrio"

    def __repr__(self):
        return f"GeoPackage({self.filepath})"

    def __str__(self):
        return f"GeoPackage at {self.filepath}"

    def lock(self, reason=None):
        # Add a filename.lock file in the same directory
        lockfile = self.filepath.with_suffix(".lock")
        logger.debug(f"Locking Geopackage {self.filepath}...")
        if reason:
            # Write file with reason
            with open(lockfile, "w") as f:
                f.write(reason)
        else:
            lockfile.touch(exist_ok=True)

    def unlock(self):
        lockfile = self.filepath.with_suffix(".lock")
        if lockfile.exists():
            logger.debug(f"Unlocking Geopackage {self.filepath}...")
            lockfile.unlink()

    def is_locked(self):
        lockfile = self.filepath.with_suffix(".lock")
        return lockfile.exists()

    def lock_reason(self) -> str:
        lockfile = self.filepath.with_suffix(".lock")
        if lockfile.exists():
            with open(lockfile, "r") as f:
                return f.read()
        return ""

    def delete_existing_file(self):
        if self.filepath.exists():
            logger.info(f"Deleting existing GeoPackage {self.filepath}...")
            self.filepath.unlink()

    def list_vector_layers(self) -> list[str]:
        return [name for name, _ in pyogrio.list_layers(self.filepath)]

    def list_raster_layers(self) -> list[str]:
        raster_layers = []
        conn = sqlite3.connect(str(self.filepath))
        cursor = conn.cursor()

        # Query to select all raster (tiles) layers from the gpkg_contents table
        # TODO add support for 3d-gridded-coverage or any other data_type
        query = "SELECT table_name FROM gpkg_contents WHERE data_type = '2d-gridded-coverage';"
        cursor.execute(query)

        rows = cursor.fetchall()
        for row in rows:
            raster_layers.append(row[0])

        conn.close()
        return raster_layers

    def list_all_layers(self) -> list[str]:
        return self.list_vector_layers() + self.list_raster_layers()

    def read_raster(self, layer_name: str) -> GeoRaster:
        with conditional_status(self.verbose) as status:
            status.update(f"Reading raster from {self.filepath}...")
            with rasterio.open(
                self.filepath, "r", driver="GPKG", table=layer_name
            ) as src:
                return GeoRaster(src.read(), src.transform, src.crs)

    def write_raster(self, raster: GeoRaster, layer_name: str) -> None:
        if not self.filepath.exists():
            raise FileNotFoundError(
                f"Output path {self.filepath} does not exist, can't write raster to non-existing GPKG."
            )

        layer_name = sanitize_layer_name(layer_name)

        # GDAL will error if the layer already exists, so we drop it
        drop_gpkg_table_if_exists(layer_name, self.filepath)

        bands, height, width = raster.raster.shape
        with rasterio.open(
            self.filepath,
            "w",
            driver="GPKG",
            width=width,
            height=height,
            count=bands,
            dtype=raster.raster.dtype,
            crs=raster.crs,
            transform=raster.transform,
            raster_table=layer_name,
            append_subdataset=True,
        ) as dst:
            dst.write(raster.raster)

    def read_vector(self, layer_name: str) -> gpd.GeoDataFrame:
        gdf = gpd.read_file(self.filepath, layer=layer_name, engine=self._engine)
        return gdf

    def write_vector(self, gdf: gpd.GeoDataFrame, layer_name: str) -> None:
        gdf.to_file(
            str(self.filepath),
            layer=layer_name,
            driver="GPKG",
            engine=self._engine,
            overwrite=True,
        )
    #
    # def write_attribute_table(self, data: pd.DataFrame, layer_name: str):
    #     # Pyogrio supports attribute only write
    #     gpd.GeoDataFrame(data).to_file(
    #         str(self.filepath), layer=layer_name, driver="GPKG", engine="pyogrio"
    #     )
    #     return
    #     ds = ogr.Open(self.filepath, update=1)
    #     if ds is None:
    #         out_driver = ogr.GetDriverByName("GPKG")
    #         ds = out_driver.CreateDataSource("local_file.gpkg")
    #
    #     if ds.GetLayer(layer_name):
    #         lyr = ds.GetLayer(layer_name)
    #     else:
    #         lyr = ds.CreateLayer(
    #             layer_name,
    #             geom_type=ogr.wkbNone,
    #             options=["ASPATIAL_VARIANT=GPKG_ATTRIBUTES"],
    #         )
    #
    #     field_names = [field.name for field in lyr.schema]
    #     for column in data.columns:
    #         if column not in field_names:
    #             if pd.api.types.is_integer_dtype(data[column]):
    #                 field_type = ogr.OFTInteger
    #             elif pd.api.types.is_float_dtype(data[column]):
    #                 field_type = ogr.OFTReal
    #             else:
    #                 field_type = ogr.OFTString  # Default to string for other types
    #             field_defn = ogr.FieldDefn(column, field_type)
    #             lyr.CreateField(field_defn)
    #
    #     for idx, row in data.iterrows():
    #         feature = ogr.Feature(lyr.GetLayerDefn())
    #         for col in data.columns:
    #             if pd.api.types.is_integer_dtype(data[col]):
    #                 feature.SetField(col, int(row[col]))
    #             elif pd.api.types.is_float_dtype(data[col]):
    #                 feature.SetField(col, float(row[col]))
    #             else:
    #                 feature.SetField(col, str(row[col]))
    #         lyr.CreateFeature(feature)
    #         feature = None
    #
    #     # Save and close the data source
    #     ds = None
    #
    # def read_attribute_table(self, layer_name: str) -> pd.DataFrame:
    #     # df = gpd.read_file(self.filepath, layer_name=layer_name, engine="pyogrio")
    #     # return df
    #     ds = ogr.Open(self.filepath)
    #     if ds is None:
    #         raise FileNotFoundError(f"No GeoPackage file found at {gpkg_file}")
    #
    #     lyr = ds.GetLayer(layer_name)
    #     if lyr is None:
    #         raise ValueError(f"Layer {layer_name} not found in the GeoPackage")
    #
    #     rows = []
    #
    #     for feature in lyr:
    #         row_data = {
    #             field.name: feature.GetField(field.name) for field in lyr.schema
    #         }
    #         rows.append(row_data)
    #
    #     df = pd.DataFrame(rows)
    #
    #     # Clean up connection
    #     ds = None
    #
    #     return df

    def drop_table(self, layer_name: str) -> None:
        raise NotImplementedError("Not implemented yet.")
        # drop_gpkg_table_if_exists(layer_name, self.filepath)

    @contextmanager
    def db_connection(self):
        conn = None
        try:
            conn = sqlite3.connect(self.filepath)
            yield conn
        except sqlite3.Error as e:
            logger.error(f"Error connecting to database: {e}")
            raise
        finally:
            if conn:
                conn.close()

    def parse_filter(self, column_filter):
        if column_filter:
            split = column_filter.split("=")
            if len(split) != 2:
                raise ValueError(
                    "Column filter must be in the form 'column=condition'."
                )
            column, condition = split
            column = column.strip()
            condition = condition.strip()
            return column, condition
        return None, None

    def get_feature_count(self, layer_name, column_filter=None):
        with self.db_connection() as conn:
            cur = conn.cursor()

            if column_filter is None:
                cur.execute(f"SELECT COUNT(*) FROM {layer_name}")
            else:
                column, condition = self.parse_filter(column_filter)
                cur.execute(
                    f"""SELECT COUNT(*) FROM {layer_name} WHERE {column} = '{condition}'"""
                )
            row_count = cur.fetchone()[0]

        return row_count

    def get_numpy_column(self, layer_name, column_name, column_filter):
        logger.info(f"Reading column {column_name} from {self.filepath}...")
        with self.db_connection() as conn:
            cur = conn.cursor()

            if column_filter is None:
                cur.execute(f"SELECT {column_name} FROM {layer_name}")
            else:
                column, condition = self.parse_filter(column_filter)
                query = f"""SELECT "{column_name}" FROM {layer_name} WHERE "{column}" = '{condition}'"""
                cur.execute(query)
            rows = cur.fetchall()

        return np.array(rows).astype(np.float64).flatten()


def list_raster_layers(filepath: PathLike) -> list[str]:
    raster_layers = []
    conn = sqlite3.connect(str(filepath))
    cursor = conn.cursor()
    # Query to select all raster (tiles) layers from the gpkg_contents table
    query = (
        "SELECT table_name FROM gpkg_contents WHERE data_type = '2d-gridded-coverage';"
    )
    cursor.execute(query)

    # Fetch all matching rows
    rows = cursor.fetchall()
    for row in rows:
        raster_layers.append(row[0])

    conn.close()
    return raster_layers


def sanitize_layer_name(layer_name: str) -> str:
    # TODO make more robust
    # only keep alphanumeric and underscore, replace with space
    sanitized = "".join(
        c if c.isalnum() or c in ("_", "-") else " " for c in layer_name
    )
    if layer_name != sanitized:
        console.log(
            f"Layer name {layer_name} contains non-alphanumeric characters, replacing with {sanitized}."
        )
    return sanitized


def drop_gpkg_table_if_exists(layer_name, output_path):
    conn = sqlite3.connect(str(output_path))
    cursor = conn.cursor()
    cursor.execute(f'DROP TABLE IF EXISTS "{layer_name}";')

    # Remove layer from gpkg_contents
    cursor.execute(f'DELETE FROM gpkg_contents WHERE table_name = "{layer_name}";')

    # If it's a raster layer, you may also need to remove entries from gpkg_tile_matrix_set and gpkg_tile_matrix
    cursor.execute(
        f'DELETE FROM gpkg_tile_matrix_set WHERE table_name = "{layer_name}";'
    )
    cursor.execute(f'DELETE FROM gpkg_tile_matrix WHERE table_name = "{layer_name}";')

    conn.commit()
    conn.close()
