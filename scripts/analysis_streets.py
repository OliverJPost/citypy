# Load the data
# Assuming your SQLite database is named 'your_database.db' and the table is 'your_table'
import sqlite3

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

conn = sqlite3.connect("/Users/ole/Downloads/global_road_edges.db")
query = "SELECT * FROM road_edges"
df = pd.read_sql_query(query, conn)
conn.close()
df = df.drop(columns=["speed_kph"])

columns = [
    "angle_entropy::nb",
    "right_neighbour_distance::nb::weighted_mean",
    "continuity::nb::weighted_median",
    "stretch_curvilinearity::nb::weighted_mean",
    "section_length::nb::mean",
    "section_length::nb::std",
    "right_neighbour_angle_deviation::nb::weighted_mean",
    "cluster_street_gmm13",
]
df = df[columns]


def calculate_cluster_zscores(df, cluster_column):
    # Separate the cluster labels from the metric columns
    clusters = df[cluster_column]
    metrics = df.drop(columns=[cluster_column])

    # Ensure only numeric columns are considered
    numeric_metrics = metrics.select_dtypes(include="number")

    # Step 1: Calculate mean and standard deviation for each metric grouped by cluster
    cluster_means = numeric_metrics.groupby(clusters).mean()
    cluster_stds = numeric_metrics.groupby(clusters).std()

    # Step 2: Calculate the global mean and standard deviation for each metric
    global_mean = numeric_metrics.mean()
    global_std = numeric_metrics.std()

    # Step 3: Calculate z-scores for each cluster, comparing cluster means to the global mean
    z_scores = (cluster_means - global_mean) / global_std

    # Step 4: Find top features with the highest absolute z-scores for each cluster
    top_features_per_cluster = z_scores.apply(
        lambda row: row.abs().nlargest(3).index.tolist(), axis=1
    )

    # Return the z-scores and top features for further analysis
    return z_scores, top_features_per_cluster


# Example usage:
# Assuming `df` is your full dataset with a column 'cluster_street_gmm3'
z_scores, top_features = calculate_cluster_zscores(df, "cluster_street_gmm13")


pd.set_option("display.max_columns", None)
# no trimming
pd.set_option("display.max_colwidth", None)
# prevent ... not showing all columns
pd.set_option("display.max_rows", None)
print(z_scores.T)
print(top_features)
