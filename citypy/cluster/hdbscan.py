import pandas as pd

# from hdbscan import HDBSCAN


def cluster_data_hdbscan(df):
    # remove rows with NaN values, but keep the original index
    df = df.dropna()
    total_count = len(df)
    # hdbscan = HDBSCAN(
    #     min_cluster_size=int(total_count * 0.007),
    #     min_samples=1,
    #     cluster_selection_epsilon=0.01,
    # )
    # hdbscan.fit(df)
    # return pd.Series(hdbscan.labels_, index=df.index)
