from pathlib import Path
from typing import Optional

import geopandas as gpd
import matplotlib.pyplot as plt
import typer

from citypy.cli import app


@app.command()
def plot(
    input_gpkg: str = typer.Argument(
        help="Path to the input GPKG file.",
    ),
    layer_name: str = typer.Argument(
        help="Name of the layer to plot.",
    ),
    column1: str = typer.Argument(
        help="Name of the column to plot on the x-axis.",
    ),
    column2: Optional[str] = typer.Argument(
        help="Name of the column to plot on the y-axis.",
    ),
    color_column: Optional[str] = typer.Argument(
        None,
        help="Name of the column to use for color.",
    ),
) -> None:
    gdf = gpd.read_file(input_gpkg, layer=layer_name, engine="pyogrio")
    if column2:
        if color_column:
            plt.scatter(gdf[column1], gdf[column2], c=gdf[color_column], alpha=0.2)
        else:
            plt.scatter(gdf[column1], gdf[column2], alpha=0.2)
        plt.xlabel(column1)
        plt.ylabel(column2)
    else:
        plt.hist(gdf[column1], alpha=0.2)
        plt.xlabel(column1)
        plt.ylabel("Frequency")
    plt.show()
