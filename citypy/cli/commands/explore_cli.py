from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import typer

from citypy.cli import app
from citypy.util.gpkg import GeoPackage


@app.command()
def explore(
    input_gpkg: str = typer.Argument(
        help="Path to the input GPKG file.",
    ),
    layer_name: str = typer.Argument(
        help="Name of the layer to plot.",
    ),
    column: str = typer.Argument(
        help="Name of the column to explore.",
    ),
) -> None:
    gpkg = GeoPackage(input_gpkg, verbose=True)
    gdf = gpkg.read_vector(layer_name)
    map = gdf.explore(column)
    tmp_dir = Path("tmp")
    outfp = r"<your dir path>\base_map.html"
    map.save(outfp)

    webbrowser.open(outfp)
