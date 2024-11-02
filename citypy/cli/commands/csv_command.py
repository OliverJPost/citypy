import geopandas as gpd
import typer

from citypy.cli import app


@app.command()
def csv(
    input_gpkg: str = typer.Argument(
        help="Path to the input GPKG file.",
    ),
    layer_name: str = typer.Argument(),
) -> None:
    gdf = gpd.read_file(input_gpkg, layer=layer_name, engine="pyogrio")
    gdf.to_csv(input_gpkg.replace(".gpkg", ".csv"), index=False)
