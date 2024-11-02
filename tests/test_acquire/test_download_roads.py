from pathlib import Path

import networkx as nx
import osmnx
import pytest
import shapely

from citypy.acquire.acquire_roads import (
    download_raw_osm_road_graph,
)
from citypy.util.gpkg import GeoPackage
from citypy.util.graph import convert_graph_to_gdfs, graph_from_nodes_edges


@pytest.fixture()
def city_name():
    return "Delfgauw, NL"


@pytest.fixture()
def raw_osm_graph() -> nx.MultiDiGraph:
    return osmnx.load_graphml("test_acquire/responses/raw_delfgauw_nl_graph.graphml")


def test_download_raw_osm_road_graph(city_name, raw_osm_graph, mocker):
    mocker.patch("osmnx.graph_from_place", return_value=raw_osm_graph.copy())
    graph = download_raw_osm_road_graph(city_name)
    assert nx.utils.graphs_equal(graph, raw_osm_graph)


def graphs_equal_verbose(g1, g2):
    # Check node set equality
    if set(g1.nodes) != set(g2.nodes):
        print("Nodes in G1 not in G2:", set(g1.nodes) - set(g2.nodes))
        print("Nodes in G2 not in G1:", set(g2.nodes) - set(g1.nodes))
        return False

    # Check edge set equality
    if set(g1.edges) != set(g2.edges):
        print("Edges in G1 not in G2:", set(g1.edges) - set(g2.edges))
        print("Edges in G2 not in G1:", set(g2.edges) - set(g1.edges))
        return False

    # Check node attributes equality
    for node in g1.nodes:
        if g1.nodes[node] != g2.nodes[node]:
            print("Node attributes differ for node:", node)
            print("G1:", g1.nodes[node])
            print("G2:", g2.nodes[node])
            return False

    # Check edge attributes equality
    for edge in g1.edges:
        edge_g1 = g1.edges[edge]
        edge_g2 = g2.edges[edge]
        # Not all OSM edges contain geometry, but all converted edges contain geometry
        # Only compare geometry if original graph has geometry
        if "geometry" in edge_g1 and edge_g1.get("geometry") != edge_g2.get("geometry"):
            print("Original edge has geometry but does not match converted edge.")
            print("G1:", edge_g1.get("geometry"))
            print("G2:", edge_g2.get("geometry"))
            return False

        edge_g1_without_geo = {k: v for k, v in edge_g1.items() if k != "geometry"}
        edge_g2_without_geo = {k: v for k, v in edge_g2.items() if k != "geometry"}
        if edge_g1_without_geo != edge_g2_without_geo:
            print("Edge attributes differ for edge:", edge)
            print("G1:", edge_g1_without_geo)
            print("G2:", edge_g2_without_geo)
            return False

    return True


def remove_edge_geometry(graph) -> None:
    for u, v, key in graph.edges(keys=True):
        if "geometry" in graph[u][v][key]:
            del graph[u][v][key]["geometry"]


def test_convert_graph_to_gdfs(raw_osm_graph):
    nodes, edges = convert_graph_to_gdfs(raw_osm_graph)
    assert len(nodes) == len(raw_osm_graph.nodes)
    assert len(edges) == len(raw_osm_graph.edges)
    for _, node in nodes.iterrows():
        assert isinstance(node["geometry"], shapely.Point)

    for _, edge in edges.iterrows():
        assert isinstance(edge["geometry"], shapely.LineString)

    converted_graph = osmnx.graph_from_gdfs(nodes, edges)
    assert graphs_equal_verbose(raw_osm_graph, converted_graph)


def test_export_import_equality(raw_osm_graph, tmp_path):
    nodes, edges = convert_graph_to_gdfs(raw_osm_graph)
    filepath = Path(tmp_path) / "temp_roads.gpkg"
    gpkg = GeoPackage(filepath)
    gpkg.write_vector(nodes, "nodes")
    gpkg.write_vector(edges, "edges")
    nodes_imported = gpkg.read_vector("nodes")
    edges_imported = gpkg.read_vector("edges")
    converted_graph = graph_from_nodes_edges(nodes_imported, edges_imported)
    assert graphs_equal_verbose(raw_osm_graph, converted_graph)
