import os
import sqlite3
from os import PathLike

import geopandas as gpd
import pyogrio
import rich
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from citypy.console import console
from citypy.util.gpkg import GeoPackage, list_raster_layers


def create_file_info_panel(filepath: PathLike) -> tuple[Panel, Panel]:
    # Calculate file size in MB
    file_size_bytes = os.path.getsize(filepath)
    file_size_mb = file_size_bytes / (1024 * 1024)
    file_size_text = f"File size: {file_size_mb:.2f} MB"

    # Splitting the filepath to highlight the filename differently
    dir_path, filename = os.path.split(os.path.abspath(filepath))
    file_path_text = Text(f"{dir_path}/", style="dim")
    file_path_text.append(filename)

    # File path (dimmed), file size, and Layers title
    text = Text.assemble(
        (filename, "bold magenta"),
        "\n",
        (f"{dir_path}/", "dim"),
        (filename, "bold"),
        "\n",
        file_size_text,
        justify="left",
    )

    # Convert layout to panel for adding the table later
    file_info_panel = Panel(text, box=rich.box.DOUBLE)

    return file_info_panel


def get_gpkg_layer_info(filepath, table_name):
    """Retrieve column names, types, and number of records from a GeoPackage table."""

    # Connect to the GeoPackage
    conn = sqlite3.connect(filepath)
    cur = conn.cursor()

    # Get column names and types
    cur.execute(f"PRAGMA table_info({table_name})")
    columns = cur.fetchall()
    column_types = {name: type for (_, name, type, _, _, _) in columns}

    # Get number of records (rows) in the table
    cur.execute(f"SELECT COUNT(*) FROM {table_name}")
    row_count = cur.fetchone()[0]

    # Close the connection
    conn.close()

    # Return both column types and row count
    return column_types, row_count


def get_raster_info(layer_name, filepath):
    conn = sqlite3.connect(str(filepath))
    cursor = conn.cursor()
    query = (
        f"SELECT data_type, srs_id FROM gpkg_contents WHERE table_name = '{layer_name}'"
    )
    cursor.execute(query)
    row = cursor.fetchall()[0]
    data_type, srs_id = row

    return data_type, f"EPSG:{srs_id}"


def display_gpkg_description(filepath: PathLike) -> None:
    # Create table for layers
    layers_table = Table(show_header=True, header_style="bold cyan")
    layers_table.add_column("Layer Name")
    layers_table.add_column("Num of Records", justify="right")
    layers_table.add_column("Columns", overflow="fold")

    gpkg = GeoPackage(filepath)
    for layer_name in gpkg.list_vector_layers():
        column_types, row_count = get_gpkg_layer_info(filepath, layer_name)
        columns_with_types = []
        for column_name, column_type in column_types.items():
            columns_with_types.append(f"{column_name} [dim]({column_type})[/dim]")
        columns_joined = ", ".join(columns_with_types)
        layers_table.add_row(layer_name, str(row_count), columns_joined)

    # Create and display file info panel
    file_info_panel = create_file_info_panel(filepath)
    console.print(file_info_panel)
    console.print(layers_table)

    raster_table = Table(show_header=True, header_style="bold cyan")
    raster_table.add_column("Layer Name")
    raster_table.add_column("Datatype")
    raster_table.add_column("Band Count")
    raster_table.add_column("Width", justify="right")
    raster_table.add_column("Height", justify="right")
    raster_table.add_column("CRS")

    for layer_name in gpkg.list_raster_layers():
        dtype, crs = get_raster_info(layer_name, filepath)
        raster_table.add_row(layer_name, dtype, "99", "999", "999", crs)

    console.print(raster_table)


def display_gpkg_layer_description(filepath: PathLike, layer_name: str) -> None:
    # Create table for layers
    layer_table = Table(show_header=True, header_style="bold cyan")
    layer_table.add_column("Column Name")
    layer_table.add_column("GPKG Type")
    layer_table.add_column("GeoPandas Type")
    layer_table.add_column("Mean", justify="right")
    layer_table.add_column("Min", justify="right")
    layer_table.add_column("Max", justify="right")
    layer_table.add_column("Non-null", justify="right")
    layer_table.add_column("Unique", overflow="fold")

    column_types, row_count = get_gpkg_layer_info(filepath, layer_name)
    data = gpd.read_file(filepath, layer=layer_name, engine="pyogrio")
    for column_name, db_column_type in column_types.items():
        if column_name in ("fid", "geom", "geometry"):
            continue
        gpd_type = str(data[column_name].dtype)
        mean = (
            f"{data[column_name].mean():.2f}"
            if gpd_type in ("float64", "int64")
            else "~"
        )
        minimum = (
            f"{data[column_name].min():.2f}"
            if gpd_type in ("float64", "int64")
            else "~"
        )
        maximum = (
            f"{data[column_name].max():.2f}"
            if gpd_type in ("float64", "int64")
            else "~"
        )
        if gpd_type == "object":
            unique_values = data[column_name].value_counts().head(7)
            unique_values = ", ".join(
                [
                    f"{value} [dim]({count})[/dim]"
                    for value, count in unique_values.items()
                ]
            )
        else:
            unique_values = "~"
        non_null_count = data[column_name].count()
        non_null_percentage = non_null_count / row_count * 100
        if non_null_percentage > 70:
            color = "green"
        elif non_null_percentage > 30:
            color = "yellow"
        else:
            color = "red"

        if non_null_percentage == 100:
            non_null = f"[green]100%"
        else:
            non_null = f"[{color}]{non_null_count} ({non_null_percentage:.2f}%)"

        layer_table.add_row(
            column_name,
            db_column_type,
            gpd_type,
            mean,
            minimum,
            maximum,
            non_null,
            unique_values,
        )

    # Create and display file info panel
    file_info_panel = create_file_info_panel(filepath)
    console.print(file_info_panel)
    console.print(f"Layer: {layer_name}")
    console.print(f"Number of records: {row_count}")
    console.print(f"CRS: {data.crs}")
    console.print(layer_table)
