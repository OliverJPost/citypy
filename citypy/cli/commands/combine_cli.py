import os
import sqlite3
from pathlib import Path
from typing import List, Optional

import fiona
import geopandas as gpd
import momepy
import numpy as np
import pandas as pd
import pyproj
import typer
from loguru import logger

from citypy.cli import app
from citypy.util.gpkg import GeoPackage
from citypy.util.graph import convert_graph_to_gdfs, graph_from_nodes_edges
from config import cfg


@app.command()
def combine(
    input_gpkg_directory: Path = typer.Argument(
        help="Path to the input GPKG file directory.", exists=True, dir_okay=True
    ),
    layer=typer.Argument(
        help="Name of the layer to combine.",
    ),
    output_db: Path = typer.Argument(
        help="Path to the output db file.",
    ),
    limit: Optional[bool] = typer.Option(
        True,
        help="Limit the number of rows to combine.",
    ),
) -> None:
    first_gpkg = input_gpkg_directory / next(
        f for f in os.listdir(input_gpkg_directory) if f.endswith(".gpkg")
    )

    with fiona.open(first_gpkg, layer=layer) as flayer:
        columns = list(flayer.schema["properties"].keys())

    quoted_columns = [f'"{col}"' for col in columns]
    columns_with_source = quoted_columns + ['"city"']

    with typer.progressbar(os.listdir(input_gpkg_directory)) as progress:
        with sqlite3.connect(output_db) as conn_out:
            cursor_out = conn_out.cursor()

            # Create a table in the new database (add an extra column for the source filename)
            statement = f"CREATE TABLE IF NOT EXISTS {layer} ({', '.join(columns_with_source)});"
            print(statement)
            cursor_out.execute(statement)

            # Iterate through all GeoPackage files in the directory
            for filename in progress:
                if not filename.endswith(".gpkg"):
                    continue

                gpkg_path = os.path.join(input_gpkg_directory, filename)
                source_file = os.path.splitext(filename)[
                    0
                ]  # Filename without extension

                # Connect to the current GeoPackage file
                with sqlite3.connect(gpkg_path) as conn_in:
                    cursor_in = conn_in.cursor()

                    # Read the "road_edges" layer, selecting only the desired columns
                    try:
                        if limit:
                            cursor_in.execute(
                                f"SELECT {', '.join(quoted_columns)} FROM {layer} ORDER BY RANDOM() LIMIT 10000;"
                            )
                        else:
                            cursor_in.execute(
                                f"SELECT {', '.join(quoted_columns)} FROM {layer};"
                            )
                    except sqlite3.OperationalError as e:
                        logger.warning(f"Skipping {filename}, reason: {e}")
                        continue

                    # Insert the data along with the source filename into the output database
                    for row in cursor_in:
                        cursor_out.execute(
                            f"INSERT INTO {layer} VALUES ({', '.join(['?'] * len(columns_with_source))});",
                            row + (source_file,),
                        )

            # Commit the changes to the output database
            conn_out.commit()
