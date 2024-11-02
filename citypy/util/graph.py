import ast

import geopandas as gpd
import networkx as nx
import osmnx
import pandas as pd
import rustworkx as rx


def convert_graph_to_gdfs(
    graph: nx.MultiDiGraph,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    nodes, edges = osmnx.graph_to_gdfs(
        graph, nodes=True, edges=True, node_geometry=True, fill_edge_geometry=True
    )
    # nodes.to_crs(nodes.estimate_utm_crs(), inplace=True)
    # edges.to_crs(edges.estimate_utm_crs(), inplace=True)
    return nodes, edges


def graph_from_nodes_edges(
    nodes: gpd.GeoDataFrame, edges: gpd.GeoDataFrame
) -> nx.Graph:
    # set index as u, v, key if not already
    if not isinstance(edges.index, pd.MultiIndex):
        edges = edges.set_index(["u", "v", "key"])

    # osmid as index if not already # FIXME use fid instead
    if "osmid" in nodes.columns:
        nodes = nodes.set_index("osmid")

    graph = osmnx.graph_from_gdfs(nodes, edges)

    # Correct for discrepancies caused by GPKG import and export
    for edge in graph.edges:
        # Convert stringified types
        for key, value in graph.edges[edge].items():
            if value == "True":
                graph.edges[edge][key] = True
            elif value == "False":
                graph.edges[edge][key] = False
            elif isinstance(value, str) and value.startswith("["):
                graph.edges[edge][key] = ast.literal_eval(value)

    return graph


def graph_from_edges_only(edges: gpd.GeoDataFrame) -> nx.Graph:
    graph_attrs = {"crs": edges.crs}
    G = nx.MultiDiGraph(**graph_attrs)

    # add edges and their attributes to graph, but filter out null attribute
    # values so that edges only get attributes with non-null values
    attr_names = edges.columns.to_list()
    for (u, v, k), attr_vals in zip(edges.index, edges.to_numpy()):
        data_all = zip(attr_names, attr_vals)
        data = {
            name: val
            for name, val in data_all
            if isinstance(val, list) or pd.notna(val)
        }
        G.add_edge(u, v, key=k, **data)

    return G


def rx_combined_graph_from_nodes_edges(
    nodes: gpd.GeoDataFrame, edges: gpd.GeoDataFrame
) -> rx.PyGraph:
    graph_attrs = {"crs": nodes.crs}
    G = rx.PyGraph()
    G.attrs = graph_attrs
    # make sure u and v are integers
    edges["u"] = edges["u"].astype(int)
    edges["v"] = edges["v"].astype(int)
    edges.set_index(["u", "v"], inplace=True)
    G.add_nodes_from(list(nodes.index))
    attr_names = edges.columns.to_list()
    for (u, v), attr_vals in zip(edges.index, edges.to_numpy()):
        data_all = zip(attr_names, attr_vals)
        data = {
            name: val
            for name, val in data_all
            if isinstance(val, list) or pd.notna(val)
        }
        try:
            G.add_edge(u, v, data)
        except IndexError:
            print(f"Error adding edge {u}, {v}")

    for node in G.nodes():
        pass

    return G
