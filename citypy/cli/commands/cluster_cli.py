from pathlib import Path
from typing import List

import geopandas as gpd
import numpy as np
import typer

from citypy.cli import app
from citypy.cluster.kmeans import cluster_data
from citypy.util.gpkg import GeoPackage


def column_automcomplete(ctx: typer.Context, incomplete: str):
    input_file = ctx.params.get("input_gpkg")
    input_layer = ctx.params.get("layer_name")
    already_selected_columns = ctx.params.get("cluster_columns")

    gdf_columns = gpd.read_file(input_file, layer=input_layer, engine="pyogrio").columns
    numeric_column_names = gdf_columns[
        gdf_columns.apply(lambda x: np.issubdtype(x, np.number))
    ]

    return [
        column
        for column in numeric_column_names
        if column not in already_selected_columns
        and column.lower().startswith(incomplete.lower())
    ]


@app.command()
def cluster(
    input_gpkg: Path = typer.Argument(
        help="Path to the input GPKG file.",
        exists=True,
    ),
    layer_name: str = typer.Argument(
        help="Name of the layer to cluster.",
    ),
    cluster_columns: List[str] = typer.Option(
        None,
        "--cluster_on",
        "--on",
        help="Name of the columns to cluster on.",
        autocompletion=column_automcomplete,
    ),
    cluster_output_column: str = typer.Option(
        "cluster", "--output_column", "-c", help="Name of the output column."
    ),
    column_filter: str = typer.Option(None, "--filter"),
) -> None:
    gpkg = GeoPackage(input_gpkg)
    gdf = gpkg.read_vector(layer_name)

    if not column_filter:
        gdf[cluster_output_column] = cluster_data(gdf[cluster_columns])
    else:
        filter_column, filter_condition = column_filter.split("=")
        filter_condition = filter_condition.strip()

        # only cluster on the columns that match the filter, set None to all others
        series = cluster_data(
            gdf[gdf[filter_column] == filter_condition][cluster_columns]
        )
        gdf[cluster_output_column] = None
        gdf.loc[gdf[filter_column] == filter_condition, cluster_output_column] = series

    # distribution_columns = {
    #     "road_edges":
    # }

    gpkg.write_vector(gdf, layer_name)
    typer.echo(f"Added column {cluster_output_column} to {layer_name} in {input_gpkg}.")
