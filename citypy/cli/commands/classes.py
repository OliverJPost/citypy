import io
from collections import defaultdict
from pathlib import Path
from typing import List
from zipfile import ZipFile

import fiona
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyproj
import seaborn as sns
import typer
from catboost import CatBoostClassifier
from loguru import logger
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import (
    accuracy_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_samples,
    silhouette_score,
)
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from citypy.cluster.kmeans import cluster_data
from citypy.util.gpkg import GeoPackage

from .. import app

not_columns = [
    "confidence",
    "geometry",
    "source",
    "id::building",
    "ghsl_height",
    "name",
    "closest_road_edge_id",
    "building_group_id",
    "enclosure_id",
    "building_class_pred",
    "building_class_pred_70",
    "building_class_pred_90",
    "ground_truth",
    "building_class",
]

categ_columns = [
    "land_use_category",
    "land_use_category::nb_radius_300::mode",
    "roof_shape",
    "subtype",
    "class",
]


def get_classification_columns(df):
    for col in categ_columns:
        df[col] = df[col].fillna("UNKNOWN")
    return df.drop(columns=not_columns, errors="ignore")


@app.command()
def classes(
    file: Path = typer.Option(
        None,
        help="Path to the input GPKG file.",
        exists=True,
    ),
    dir: Path = typer.Option(
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
    classes_output_file: Path = typer.Option(
        None, "--output", "-o", help="Name of the output file. Suffix is ignored."
    ),
) -> None:
    if file is None and dir is None:
        raise typer.BadParameter("Either --file or --dir must be provided.")
    if file is not None and dir is not None:
        raise typer.BadParameter("Only one of --file or --dir can be provided.")

    files = []
    if file:
        files.append(file)
    if dir:
        files.extend(dir.glob("*.gpkg"))

    final_files = []
    for input_gpkg in tqdm(files):
        gpkg = GeoPackage(input_gpkg)
        if gpkg.is_locked():
            logger.warning(f"Skipping {input_gpkg} because it is locked.")

        with fiona.open(input_gpkg, layer="buildings") as layer:
            columns = layer.schema["properties"].keys()
            if "ground_truth" not in columns:
                logger.warning(
                    f"Skipping {input_gpkg} because it does not have a ground_truth layer."
                )
                continue

        final_files.append(input_gpkg)

    all_labels = pd.Series()
    all_features = pd.DataFrame()

    for input_gpkg in tqdm(final_files):
        gpkg = GeoPackage(input_gpkg)
        logger.info(f"Reading {input_gpkg}")
        buildings = gpkg.read_vector("buildings")
        labeled_buildings = buildings[buildings["ground_truth"].notna()]
        labeled_buildings = labeled_buildings[
            ~labeled_buildings["ground_truth"].isin(
                ["outbuilding", "invalid", "semi_detached", "agriculture", "tower"]
            )
        ]
        try:
            gdf_base = gpd.read_file("ground_truth.gpkg")
        except:
            gdf_base = gpd.GeoDataFrame(
                columns=["geometry"],
                geometry="geometry",
                crs=pyproj.CRS.from_epsg(4326),
            )

        gdf_base = gpd.GeoDataFrame(
            pd.concat([gdf_base, labeled_buildings.to_crs(4326)], ignore_index=True),
            crs=gdf_base.crs,
        )
        gdf_base.to_file("ground_truth.gpkg", layer="layer1", driver="GPKG")

        labels = labeled_buildings["ground_truth"]
        train_data = get_classification_columns(labeled_buildings)
        all_labels = pd.concat([all_labels, labels])
        all_features = pd.concat([all_features, train_data])

    logger.info("Classifying data")
    model = CatBoostClassifier(
        iterations=1000,  # Number of boosting iterations
        learning_rate=0.1,  # Learning rate
        depth=6,  # Depth of the tree
        loss_function="MultiClass",  # Specify multiclass classification
        verbose=True,  # Enable verbose output for tracking the training process
        cat_features=categ_columns,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        all_features, all_labels, test_size=0.2, random_state=42
    )

    non_numeric_columns = X_train.select_dtypes(exclude=[np.number]).columns
    print("NON NUMERIC")
    print(non_numeric_columns)

    model.fit(
        X_train,
        y_train,
        cat_features=categ_columns,
    )

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}")

    re = model.get_feature_importance(prettified=True)
    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_colwidth", None)
    print(re)

    model.save_model(classes_output_file, format="cbm")
    #
    # # Save column names and serialized model to zip file
    # with ZipFile(classes_output_file.with_suffix(".catboost"), "w") as z:
    #     z.writestr("columns.txt", "\n".join(cluster_columns))
    #     # Use numpy save to serialize and write the data to the zip
    #     with z.open("model.bin", "w") as f:
    #         buffer = io.BytesIO()
    #         np.save(
    #             buffer, cluster_centroids
    #         )  # serialize the centroids to buffer using numpy
    #         f.write(buffer.getvalue())  # write the buffer content to the file
