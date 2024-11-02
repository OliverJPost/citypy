import io
import pickle
from pathlib import Path
from zipfile import ZipFile

import fiona
import numpy as np
import typer
from loguru import logger
from sklearn.cluster import KMeans
from tqdm import tqdm

from citypy.cli.commands.clusters import cluster_app
from citypy.util.gpkg import GeoPackage


@cluster_app.command()
def apply(
    file: Path = typer.Option(
        None,
        help="Path to the input GPKG file.",
        exists=True,
    ),
    directory: Path = typer.Option(
        None,
        help="Path to a folder with input GPKG files.",
        exists=True,
    ),
    layer_name: str = typer.Option(
        None,
        "--layer",
        "-l",
        help="Name of the layer to cluster.",
    ),
    cluster_column_names: list[str] = typer.Option(
        None,
        "--column",
        "--col",
        help="Name of the output column.",
    ),
    cluster_files: list[Path] = typer.Option(
        None,
        "--clusters",
        "--cluster-file",
        help="Path to the cluster file.",
        exists=True,
    ),
    column_filter: str = typer.Option(
        None,
        help="Filter the layer before applying the clustering.",
    ),
):
    if len(cluster_column_names) != len(cluster_files):
        raise ValueError(
            f"Number of cluster column names ({len(cluster_column_names)}) must"
            f" match the number of cluster files ({len(cluster_files)})"
        )

    clustering_sets = []
    for column, cluster_file in zip(cluster_column_names, cluster_files):
        cluster_extention = cluster_file.suffix

        # cluster_file is a zip composed of a list of column names and a serialized clustering model
        with ZipFile(cluster_file) as z:
            cluster_columns = z.read("columns.txt").decode().split("\n")
            bin_model = z.read("model.bin")

        if cluster_extention == ".kmeans":
            buffer = io.BytesIO(bin_model)
            cluster_centroids_array = np.load(
                buffer
            )  # load the centroids from the buffer
            n_clusters = cluster_centroids_array.shape[0]
            model = KMeans(n_clusters=n_clusters)
            dummy_data = np.random.rand(100, len(cluster_columns))
            model.fit(dummy_data)
            model.cluster_centers_ = cluster_centroids_array
        elif cluster_extention == ".gmm":
            buffer = io.BytesIO(bin_model)
            model = pickle.load(buffer)
        else:
            raise ValueError(f"Unknown cluster file extension {cluster_extention}")

        clustering_sets.append((column, model))

    files = [file] if file else directory.glob("*.gpkg")
    for file in tqdm(files):
        gpkg = GeoPackage(file)
        with fiona.open(file, layer="road_edges") as layer:
            columns = layer.schema["properties"].keys()
            if "cluster_street_gmm13" in columns:
                logger.warning(
                    f"Skipping {file} because it already has clustering columns."
                )
                continue

        gdf = gpkg.read_vector(layer_name)
        if column_filter:
            filter_column, filter_condition = column_filter.split("=")
            filtered_gdf = gdf[gdf[filter_column] == filter_condition][
                cluster_columns
            ].dropna()
        else:
            filtered_gdf = gdf[cluster_columns].dropna()

        features = filtered_gdf.to_numpy()
        # Normalise to 0.0 to 1.0
        features_normalised = (features - features.min(axis=0)) / (
            features.max(axis=0) - features.min(axis=0)
        )

        for cluster_column_name, model in clustering_sets:
            labels = model.predict(features_normalised)
            # Apply back, filling na rows with None
            gdf[cluster_column_name] = None
            # Assign labels only to valid rows
            gdf.loc[filtered_gdf.index, cluster_column_name] = labels

        gpkg.write_vector(gdf, layer_name)
        logger.info(
            f"Applied clustering to {file} layer {layer_name} with {cluster_column_name} column."
        )
