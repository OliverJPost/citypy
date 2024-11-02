from typing import List, Optional

import geopandas as gpd
import numpy as np
import typer

from citypy.cli import app


@app.command()
def normalise(
    input_gpkg: str = typer.Argument(
        help="Path to the input GPKG file.",
    ),
    output_gpkg: Optional[str] = typer.Argument(
        None,
        help="Path to the output GPKG file.",
    ),
) -> None:
    input_gpkg = input_gpkg.strip()
    output_gpkg = output_gpkg.strip() if output_gpkg else input_gpkg

    raise NotImplementedError("This function is not implemented yet.")
    # city = CityData.from_gpkg(input_gpkg, verbose=True)
    # for layer_name in [layer_name for layer_name in city.__dict__.keys()]:
    #     if layer_name.endswith("_normalised"):
    #         continue
    #     if isinstance(getattr(city, layer_name), gpd.GeoDataFrame):
    #         gdf = getattr(city, layer_name)
    #         gdf = normalize_and_clip(gdf)
    #
    #         city.__setattr__(layer_name + "_normalised", gdf)
    #
    # city.to_gpkg(output_gpkg)


def normalize_and_clip(df):
    # Function to normalize a single column
    def normalize_column(column):
        # Calculate the 0th and 99th percentiles
        p0 = column.quantile(0)
        p99 = column.quantile(0.99)

        # Avoid division by zero in case p0 == p99
        if p0 == p99:
            return column.clip(lower=p0, upper=p99)

        # Normalize between 0 and 1
        normalized_column = (column - p0) / (p99 - p0)

        # Clip values above 1.0
        return normalized_column.clip(upper=1.0)

    # Apply normalization and clipping to numeric columns only
    for column_name in df.select_dtypes(include=np.number).columns:
        # TODO namespace ids
        if column_name.lower().endswith("id"):
            continue
        df[column_name] = normalize_column(df[column_name])

    return df
