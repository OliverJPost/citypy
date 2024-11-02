import pandas as pd
from sklearn.cluster import KMeans


def cluster_data(df, prefix=None):
    # remove rows with NaN values, but keep the original index
    df = df.dropna()

    # normalise per column
    for column in df.columns:
        df[column] = (df[column] - df[column].mean()) / df[column].std()

    cluster_count = 8
    kmeans = KMeans(n_clusters=cluster_count)
    kmeans.fit(df)
    result = pd.Series(kmeans.predict(df), index=df.index)
    if prefix:
        # make into string with prefix_ before it
        result = result.apply(lambda x: f"{prefix}_{x}")
    return result
