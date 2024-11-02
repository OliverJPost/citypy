from typing import Optional

import typer

from citypy import __app_name__, __version__
from citypy.cli import app
from citypy.logging_setup import logger


def _version_callback(value: bool) -> None:  # noqa: FBT001
    if value:
        typer.echo(f"{__app_name__} v{__version__}")
        raise typer.Exit()  # noqa: RSE102


@app.callback()
def main(
    version: Optional[bool] = typer.Option(  # noqa: UP007
        None,
        "--version",
        "-v",
        help="Show the application's version and exit.",
        callback=_version_callback,
        is_eager=True,
    ),
) -> None:
    logger.info("Starting citypy")
    _ = version  # what to do with this
