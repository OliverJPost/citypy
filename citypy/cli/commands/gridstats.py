import time
from typing import Optional

import geopandas as gpd
import pandas as pd
import pylandstats as pls
import seaborn as sns
import typer
from matplotlib import pyplot as plt
from rasterio.features import rasterize
from rasterio.transform import from_origin

from citypy.cli import app
from config import cfg


@app.command()
def gridstats(
    input_gpkg: str = typer.Argument(
        help="Path to the input GPKG file.",
    ),
    column_name: str = typer.Argument(
        help="Name of the cluster column to calculate statistics for.",
    ),
) -> None:
    grid = gpd.read_file(input_gpkg, layer=cfg.layers.grid.name)

    bounds = grid.total_bounds
    width, height = (
        grid.geometry.apply(lambda geom: geom.bounds[2] - geom.bounds[0]).mean(),
        grid.geometry.apply(lambda geom: geom.bounds[3] - geom.bounds[1]).mean(),
    )
    rows = int((bounds[3] - bounds[1]) / height)
    cols = int((bounds[2] - bounds[0]) / width)

    cell_size = width
    transform = from_origin(bounds[0], bounds[3], cell_size, cell_size)

    if grid[column_name].dtype == object:
        cleaned_column = grid[column_name].fillna(
            "NaN"
        )  # Replace NaN with a placeholder string
        unique_values, unique_indices = pd.factorize(cleaned_column)
        value_map = dict(zip(cleaned_column, unique_values))
    else:
        value_map = None

    # Create shapes for rasterization, using the mapping if necessary
    shapes = [
        (
            geom,
            value_map[value]
            if value_map and value is not None
            else int(value)
            if value is not None and not pd.isna(value)
            else None,
        )
        for geom, value in zip(grid.geometry, grid[column_name])
    ]

    raster = rasterize(
        shapes=shapes,
        out_shape=(rows, cols),
        transform=transform,
    )

    landscape = pls.Landscape(raster, res=(100, 100), nodata=0)
    patches = landscape.compute_patch_metrics_df()
    landscape.plot_landscape()
    print(patches)
    classes = landscape.compute_class_metrics_df()
    print("---" * 10)
    print("Classes")
    print(classes.T)

    plot(column_name, "distance_to_city_center", grid)
    plot(column_name, "distance_to_highway", grid)
    plot(column_name, "distance_to_primary", grid)
    plot(column_name, "distance_to_secondary", grid)


def plot(column_name, y, grid):
    sns.set(style="whitegrid")
    plt.figure(figsize=(12, 8))
    violin_plot = sns.violinplot(
        x=column_name,
        y=y,
        data=grid,
        palette="viridis",
        inner="quartile",
    )
    violin_plot.set_title(f"Density and Distribution of {y} by Class")
    violin_plot.set_xlabel("Class")
    violin_plot.set_ylabel(f"{y} (meters)")
    plt.show()
