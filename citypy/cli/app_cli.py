import typer

from .commands.clusters import cluster_app

app = typer.Typer(pretty_exceptions_enable=False)
app.add_typer(cluster_app, name="clusters")
