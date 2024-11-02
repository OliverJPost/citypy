import colorsys
from pathlib import Path
from typing import Annotated, List

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import typer
from rasterio.features import rasterize
from rasterio.transform import from_origin

from citypy.cli import app
from config import cfg


def get_distinct_colors(n):
    """Generate visually distinct colors in RGB normalized to 0-255."""
    colors = []
    for i in range(n):
        hue = i / n
        saturation = 0.75  # Reduced for slightly less intense colors
        value = 0.95  # High value for brightness
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        colors.append(tuple(int(c * 255) for c in rgb))
    return colors


FIXED_COLORS = {
    "industrial": (255, 0, 0),
    "residential": (0, 255, 0),
    "commercial": (0, 0, 255),
    "office": (255, 255, 0),
    "retail": (255, 0, 255),
}


@app.command()
def rasterify(
    input_gpkg: Annotated[
        Path,
        typer.Argument(
            help="Path to the input GPKG file.",
            exists=True,
        ),
    ],
    output_raster: Annotated[
        Path,
        typer.Argument(
            help="Path to the output raster file.",
        ),
    ],
    column_names: Annotated[
        List[str],
        typer.Argument(
            help="Name of the column on the grid layer to rasterize.",
        ),
    ],
    categorical: bool = typer.Option(
        False,
        "--categorical",
        help="Rasterize a single column as categorical data with distinct RGB values.",
    ),
):
    grid = gpd.read_file(input_gpkg, layer=cfg.layers.grid.name, engine="pyogrio")

    bounds = grid.total_bounds
    width, height = (
        grid.geometry.apply(lambda geom: geom.bounds[2] - geom.bounds[0]).mean(),
        grid.geometry.apply(lambda geom: geom.bounds[3] - geom.bounds[1]).mean(),
    )
    rows = int((bounds[3] - bounds[1]) / height)
    cols = int((bounds[2] - bounds[0]) / width)

    cell_size = width
    transform = from_origin(bounds[0], bounds[3], cell_size, cell_size)

    meta = {
        "driver": "GTiff",
        "height": rows,
        "width": cols,
        "count": len(column_names),
        "dtype": rasterio.uint8,
        "crs": grid.crs,
        "transform": transform,
    }

    if categorical:
        if len(column_names) != 1:
            typer.echo(
                "Error: Only one column should be provided for categorical rasterization."
            )
            raise typer.Exit()

        column_name = column_names[0]
        unique_values = grid[column_name].dropna().unique()
        color_map = {
            val: color
            for val, color in zip(
                unique_values, get_distinct_colors(len(unique_values))
            )
        }

        meta.update(dtype="uint8", count=3)  # Three bands for RGB

        with rasterio.open(output_raster, "w", **meta) as out_raster:
            shapes_r = [
                (
                    geom,
                    FIXED_COLORS[value][0]
                    if value in FIXED_COLORS
                    else color_map[value][0]
                    if pd.notna(value)
                    else 0,
                )
                for geom, value in zip(grid.geometry, grid[column_name])
            ]
            shapes_g = [
                (
                    geom,
                    FIXED_COLORS[value][1]
                    if value in FIXED_COLORS
                    else color_map[value][1]
                    if pd.notna(value)
                    else 0,
                )
                for geom, value in zip(grid.geometry, grid[column_name])
            ]
            shapes_b = [
                (
                    geom,
                    FIXED_COLORS[value][2]
                    if value in FIXED_COLORS
                    else color_map[value][2]
                    if pd.notna(value)
                    else 0,
                )
                for geom, value in zip(grid.geometry, grid[column_name])
            ]

            # Rasterize each band separately
            raster_r = rasterize(
                shapes=shapes_r, out_shape=(rows, cols), transform=transform, fill=0
            )
            raster_g = rasterize(
                shapes=shapes_g, out_shape=(rows, cols), transform=transform, fill=0
            )
            raster_b = rasterize(
                shapes=shapes_b, out_shape=(rows, cols), transform=transform, fill=0
            )

            # Write each band to the raster file
            out_raster.write_band(1, raster_r)
            out_raster.write_band(2, raster_g)
            out_raster.write_band(3, raster_b)

    else:
        with rasterio.open(output_raster, "w", **meta) as out_raster:
            for i, column_name in enumerate(column_names):
                if (
                    grid[column_name].dtype == np.float32
                    or grid[column_name].dtype == np.float64
                ):
                    grid[column_name] = (
                        grid[column_name] - grid[column_name].min()
                    ) / (grid[column_name].max() - grid[column_name].min())

                    # take to 0-255
                    grid[column_name] = grid[column_name] * 255

                shapes = [
                    (
                        geom,
                        255
                        if value is True
                        else 0
                        if value is False
                        else 99
                        if value is None
                        else int(float(value)),
                    )
                    for geom, value in zip(grid.geometry, grid[column_name])
                ]
                out_raster.write_band(
                    i + 1,
                    rasterize(
                        shapes=shapes, out_shape=(rows, cols), transform=transform
                    ),
                )
