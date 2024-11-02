from typing import Optional

import typer

from citypy.cli import app
from citypy.describe import display_gpkg_description, display_gpkg_layer_description


@app.command()
def describe(
    gpkg: str = typer.Argument(
        help="Path to the GPKG file.",
    ),
    layer: Optional[str] = typer.Argument(
        None,
        help="Name of the layer to describe.",
    ),
) -> None:
    if layer:
        display_gpkg_layer_description(gpkg, layer)
    else:
        display_gpkg_description(gpkg)
