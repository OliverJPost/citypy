import io
import pickle
from collections import defaultdict
from pathlib import Path
from typing import List
from zipfile import ZipFile

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import typer
from clustergram import Clustergram
from loguru import logger
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_samples,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture
from tqdm import tqdm

from citypy.cluster.kmeans import cluster_data
from citypy.util.gpkg import GeoPackage

from . import cluster_app


def plot_cluster_size_distribution(labels, fn):
    cluster_sizes = np.bincount(labels)
    plt.figure(figsize=(8, 6))
    sns.barplot(x=np.arange(len(cluster_sizes)), y=cluster_sizes, palette="viridis")
    plt.title("Cluster Size Distribution")
    plt.xlabel("Cluster Label")
    plt.ylabel("Number of Points")
    plt.savefig(fn)
    plt.close()


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


def plot_tsne(X, labels, fn):
    tsne = TSNE(n_components=2, perplexity=30, n_iter=300)
    X_tsne = tsne.fit_transform(X)

    plt.figure(figsize=(8, 6))
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=labels, cmap="viridis", s=50)
    plt.title("t-SNE Cluster Plot")
    plt.savefig(fn)
    plt.close()


def plot_silhouette(X, silhouette_avg, labels, fn):
    sample_silhouette_values = silhouette_samples(X, labels)
    y_lower = 10
    n_clusters = len(np.unique(labels))

    plt.figure(figsize=(8, 6))

    for i in range(n_clusters):
        ith_cluster_silhouette_values = sample_silhouette_values[labels == i]
        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = plt.cm.Spectral(float(i) / n_clusters)
        plt.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            ith_cluster_silhouette_values,
            facecolor=color,
            edgecolor=color,
            alpha=0.7,
        )

        plt.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
        y_lower = y_upper + 10

    plt.axvline(x=silhouette_avg, color="red", linestyle="--")
    plt.title("Silhouette Plot for each cluster")
    plt.xlabel("Silhouette coefficient values")
    plt.ylabel("Cluster label")
    plt.savefig(fn)
    plt.close()


@cluster_app.command()
def compute(
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
    kmeans_cluster_counts: list[int] = typer.Option(
        None,
        "--kmeans-clusters",
        "-k",
        help="Number of clusters to create.",
    ),
    layer_name: str = typer.Option(
        None,
        "--layer",
        "-l",
        help="Name of the layer to cluster.",
    ),
    cluster_columns: List[str] = typer.Option(
        None,
        "--cluster_on",
        "--on",
        help="Name of the columns to cluster on.",
        autocompletion=column_automcomplete,
    ),
    cluster_output_files: list[Path] = typer.Option(
        None, "--output", "-o", help="Name of the output file. Suffix is ignored."
    ),
    column_filter: str = typer.Option(None, "--filter"),
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

    total_feature_len = 0
    for input_gpkg in tqdm(files):
        gpkg = GeoPackage(input_gpkg)
        if gpkg.is_locked():
            logger.warning(f"Skipping {input_gpkg} because it is locked.")
            continue

        total_feature_len += gpkg.get_feature_count(
            layer_name, column_filter=column_filter
        )

    features = np.empty((total_feature_len, len(cluster_columns)), dtype=np.float64)

    current_idx = 0
    for input_gpkg in tqdm(files):
        gpkg = GeoPackage(input_gpkg)
        if gpkg.is_locked():
            logger.warning(f"Skipping {input_gpkg} because it is locked.")
            continue
        feature_count = gpkg.get_feature_count(layer_name, column_filter=column_filter)
        for column_idx, column in enumerate(cluster_columns):
            column_data = gpkg.get_numpy_column(
                layer_name, column, column_filter=column_filter
            )
            features[current_idx : current_idx + feature_count, column_idx] = (
                column_data
            )
        current_idx += feature_count

    # Error if any of the columns is only 0s
    if np.all(features == 0, axis=0).any():
        # identify 0 column
        zero_columns = np.where(np.all(features == 0, axis=0))[0]
        for zero_column in zero_columns:
            logger.error(
                f"Column {cluster_columns[zero_column]} contains only 0s. Please remove it."
            )
        raise ValueError("One or more columns contain only 0s.")

    # Drop any rows containing nan
    features = features[~np.isnan(features).any(axis=1)]

    # if "section_length::nb::std" in columns, clip that column to 300, to prevent outliers influencing normalisation
    if "section_length::nb::std" in cluster_columns:
        features[:, cluster_columns.index("section_length::nb::std")] = np.clip(
            features[:, cluster_columns.index("section_length::nb::std")], 0, 300
        )

    # if "right_neighbour_angle_deviation::nb::weighted_mean" in columns, clip that column to 40, to prevent outliers
    if "right_neighbour_angle_deviation::nb::weighted_mean" in cluster_columns:
        features[
            :,
            cluster_columns.index("right_neighbour_angle_deviation::nb::weighted_mean"),
        ] = np.clip(
            features[
                :,
                cluster_columns.index(
                    "right_neighbour_angle_deviation::nb::weighted_mean"
                ),
            ],
            0,
            40,
        )

    # Normalise every column to 0.0 to 1.0 range
    features = (features - features.min(axis=0)) / (
        features.max(axis=0) - features.min(axis=0)
    )

    metrics = defaultdict(dict)
    logger.info(
        f"Clustering {features.shape[0]} features with {features.shape[1]} columns."
    )
    # For every column, print percentage 0.0 and mean
    for column_idx, column in enumerate(cluster_columns):
        column_data = features[:, column_idx]
        logger.info(
            f"{column}: {np.count_nonzero(column_data == 0.0) / column_data.shape[0] * 100:.2f}% 0.0, {column_data.mean():.2f} mean"
        )
    #
    # cgram = Clustergram(range(4, 16), method="gmm")
    # cgram.fit(features)
    # print("Fitting complete")
    # cgram.plot()
    # plt.savefig("clustergram.png")

    # decimate features to subset by nth feature, n is 20
    features = features[::20]

    bic = []
    for i in range(6, 19):
        logger.info(f"Computing BIC for {i} components")
        bic_sep = 0
        gmm = GaussianMixture(n_components=i, random_state=42)
        gmm.fit(features)
        bic_sep += gmm.bic(features)
        bic.append(bic_sep / 4)
        logger.info(f"BIC for {i} components: {bic[-1]}")
        with ZipFile(f"gmm_{i}.gmm", "w") as z:
            z.writestr("columns.txt", "\n".join(cluster_columns))
            with z.open("model.bin", "w") as f:
                pickle.dump(gmm, f)

        # Step 2: Extract the means of the Gaussian components
        gmm_means = gmm.means_

        # Step 3: Apply hierarchical clustering to the GMM component means
        linked = linkage(gmm_means, method="ward")

        # Step 4: Plot the dendrogram
        plt.figure(figsize=(8, 5))
        dendrogram(linked, labels=[f"Component {t}" for t in range(len(gmm_means))])
        plt.title("Dendrogram of GMM Component Means")
        plt.xlabel("GMM Components")
        plt.ylabel("Euclidean Distance")
        plt.savefig(f"dendrogram_{i}.png")

    plt.figure(figsize=(8, 5))
    plt.plot(range(4, 22), bic, marker="o")
    plt.title("BIC for Gaussian Mixture Models")
    plt.xlabel("Number of Components")
    plt.ylabel("BIC")
    plt.savefig("bic.png")

    bic_gradient = bic_gradient = np.gradient(bic, range(4, 22))
    plt.figure(figsize=(8, 5))
    plt.plot(range(4, 22), bic_gradient, marker="o")
    plt.title("BIC Gradient for Gaussian Mixture Models")
    plt.xlabel("Number of Components")
    plt.ylabel("BIC Gradient")
    plt.savefig("bic_gradient.png")

    for i in range(6, 19):
        logger.info(f"Computing BIC for {i} components")
        bic_sep = 0
        for j in range(3):
            logger.info(f"Itteration {j}")
            gmm = GaussianMixture(n_components=i, random_state=j)
            gmm.fit(features)
            bic_sep += gmm.bic(features)
        bic_idx = i - 6
        old_bic = bic[bic_idx]
        bic[bic_idx] = (bic_sep + old_bic) / 4
        logger.info(f"BIC for {i} components: {bic[-1]}")

    plt.figure(figsize=(8, 5))
    plt.plot(range(4, 22), bic, marker="o")
    plt.title("BIC for Gaussian Mixture Models")
    plt.xlabel("Number of Components")
    plt.ylabel("BIC")
    plt.savefig("bic.png")

    bic_gradient = bic_gradient = np.gradient(bic, range(4, 22))
    plt.figure(figsize=(8, 5))
    plt.plot(range(4, 22), bic_gradient, marker="o")
    plt.title("BIC Gradient for Gaussian Mixture Models")
    plt.xlabel("Number of Components")
    plt.ylabel("BIC Gradient")
    plt.savefig("bic_gradient.png")

    return
    for kmeans_cluster_count, cluster_output_file in zip(
        kmeans_cluster_counts, cluster_output_files, strict=True
    ):
        logger.info(f"Clustering with {kmeans_cluster_count} clusters.")
        kmeans = KMeans(n_clusters=kmeans_cluster_count, copy_x=False)
        kmeans.fit(features)
        logger.info(f"KMeans clustering complete with {kmeans_cluster_count} clusters.")
        logger.info(f"Clustering inertia is {kmeans.inertia_}")

        feature_subset = features[
            np.random.choice(features.shape[0], 10000, replace=False)
        ]
        labels = kmeans.predict(feature_subset)
        # plot pca of labels
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(feature_subset)
        metrics[kmeans_cluster_count]["intertia"] = kmeans.inertia_
        metrics[kmeans_cluster_count]["silhouette"] = silhouette_score(
            feature_subset, labels
        )
        metrics[kmeans_cluster_count]["calinski-harabasz"] = calinski_harabasz_score(
            feature_subset, labels
        )
        metrics[kmeans_cluster_count]["davies-bouldin"] = davies_bouldin_score(
            feature_subset, labels
        )
        with open(cluster_output_file.with_suffix(".metrics.txt"), "w") as f:
            txt = f"KMeans Clustering Metrics for {kmeans_cluster_count} clusters\n"
            txt += f"Silhouette Score: {metrics[kmeans_cluster_count]['silhouette']}\n"
            txt += f"Calinski-Harabasz Score: {metrics[kmeans_cluster_count]['calinski-harabasz']}\n"
            txt += f"Davies-Bouldin Score: {metrics[kmeans_cluster_count]['davies-bouldin']}\n"
            f.write(txt)

        plot_silhouette(
            feature_subset,
            metrics[kmeans_cluster_count]["silhouette"],
            labels,
            cluster_output_file.with_suffix(".silhouette.png"),
        )

        for column_name in cluster_columns:
            # add subset data per label
            metrics[kmeans_cluster_count][column_name] = {}
            for cluster in np.unique(labels):
                metrics[kmeans_cluster_count][column_name][cluster] = feature_subset[
                    labels == cluster, cluster_columns.index(column_name)
                ]

        plot_cluster_size_distribution(
            labels, cluster_output_file.with_suffix(".cluster_size.png")
        )
        plot_tsne(feature_subset, labels, cluster_output_file.with_suffix(".tnse.png"))

        plt.scatter(pca_result[:, 0], pca_result[:, 1], c=labels, cmap="viridis")
        plt.title("PCA of KMeans labels")
        plt.savefig(cluster_output_file.with_suffix(".pca.png"))

        # Save cluster centroids to custom file
        cluster_centroids = kmeans.cluster_centers_

        # Save column names and serialized model to zip file
        with ZipFile(cluster_output_file.with_suffix(".kmeans"), "w") as z:
            z.writestr("columns.txt", "\n".join(cluster_columns))
            # Use numpy save to serialize and write the data to the zip
            with z.open("model.bin", "w") as f:
                buffer = io.BytesIO()
                np.save(
                    buffer, cluster_centroids
                )  # serialize the centroids to buffer using numpy
                f.write(buffer.getvalue())  # write the buffer content to the file

        # boxplot per cluster column with every cluster
        for column_name in cluster_columns:
            plt.figure(figsize=(8, 6))
            column_data = {}
            for cluster in np.unique(labels):
                column_data[cluster] = metrics[kmeans_cluster_count][column_name][
                    cluster
                ]
            sns.boxplot(data=list(column_data.values()), palette="viridis")
            plt.title(f"{column_name} boxplot per cluster")
            plt.savefig(cluster_output_file.with_suffix(f".{column_name}_boxplot.png"))
            plt.close()

    inertia_labels, inertia_scores = zip(*metrics.items())
    inertia_scores = [score["intertia"] for score in inertia_scores]

    plt.figure(figsize=(8, 6))
    plt.plot(inertia_labels, inertia_scores, marker="o")
    plt.title("Elbow Method For Optimal Clusters")
    plt.xlabel("Number of clusters")
    plt.ylabel("Inertia")
    plt.savefig("elbow_method.png")
