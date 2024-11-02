import networkx as nx
import pytest

from citypy.contextualize.road_neighborhood import (
    aggregate_edge_neighborhoods,
    find_edge_neighborhood,
    get_edge_class,
    get_neighbor_edges,
)
from citypy.road_class import EdgeClass


@pytest.fixture()
def simple_graph():
    G = nx.MultiDiGraph()
    highway = {"highway": "motorway"}
    major = {"highway": "primary"}
    street = {"highway": "residential"}
    G.add_edge(10, 5, 0, **major)
    G.add_edge(5, 10, 0, **major)
    G.add_edge(10, 9, 0, **street)
    G.add_edge(9, 10, 0, **street)
    G.add_edge(9, 8, 0, **major)
    G.add_edge(8, 9, 0, **major)
    G.add_edge(8, 6, 0, **street)
    G.add_edge(6, 8, 0, **street)
    G.add_edge(5, 6, 0, **highway)
    G.add_edge(6, 5, 0, **highway)
    G.add_edge(6, 7, 0, **highway)
    G.add_edge(7, 6, 0, **highway)
    G.add_edge(10, 11, 0, **major)
    G.add_edge(11, 10, 0, **major)
    G.add_edge(11, 12, 0, **major)
    G.add_edge(12, 11, 0, **major)
    G.add_edge(11, 2, 0, **street)
    G.add_edge(5, 2, 0, **highway)
    G.add_edge(2, 5, 0, **highway)
    G.add_edge(1, 2, 0, **highway)
    G.add_edge(2, 1, 0, **highway)
    G.add_edge(3, 2, 0, **street)
    G.add_edge(3, 5, 0, **street)
    G.add_edge(4, 3, 0, **street)
    G.add_edge(3, 4, 0, **street)
    G.add_edge(4, 13, 0, **street)
    G.add_edge(13, 4, 0, **street)
    G.add_edge(14, 13, 0, **street)
    G.add_edge(13, 14, 0, **street)
    # multiedge
    G.add_edge(13, 4, 1, **highway)

    return G


def test_get_neighbor_edges(simple_graph):
    assert get_neighbor_edges(simple_graph, (5, 10, 0)) == {
        (10, 5, 0),
        (2, 5, 0),
        (5, 2, 0),
        (5, 6, 0),
        (6, 5, 0),
        (3, 5, 0),
        (10, 9, 0),
        (9, 10, 0),
        (10, 11, 0),
        (11, 10, 0),
    }


def test_get_edge_class(simple_graph):
    assert get_edge_class(simple_graph, (5, 10, 0)) == EdgeClass.MAJOR
    assert get_edge_class(simple_graph, (10, 5, 0)) == EdgeClass.MAJOR
    assert get_edge_class(simple_graph, (5, 6, 0)) == EdgeClass.HIGHWAY
    assert get_edge_class(simple_graph, (6, 5, 0)) == EdgeClass.HIGHWAY
    assert get_edge_class(simple_graph, (13, 4, 0)) == EdgeClass.STREET
    assert get_edge_class(simple_graph, (13, 4, 1)) == EdgeClass.HIGHWAY


def test_find_edge_neighborhood(simple_graph):
    nb_edges, nb_nodes = find_edge_neighborhood(
        simple_graph, (5, 10, 0), 2, grow_to_lower_class=True
    )
    assert set(nb_edges) == {
        (3, 2, 0),
        (3, 5, 0),
        (11, 2, 0),
        (1, 2, 0),
        (2, 1, 0),
        (9, 10, 0),
        (10, 9, 0),
        (8, 9, 0),
        (9, 8, 0),
        (6, 8, 0),
        (8, 6, 0),
        (6, 5, 0),
        (5, 6, 0),
        (6, 7, 0),
        (7, 6, 0),
        (10, 11, 0),
        (11, 10, 0),
        (11, 12, 0),
        (12, 11, 0),
        (3, 4, 0),
        (10, 5, 0),
        (5, 10, 0),
        (2, 5, 0),
        (4, 3, 0),
        (5, 2, 0),
    }


def test_find_edge_neighborhood_major(simple_graph):
    nb_edges, nb_nodes = find_edge_neighborhood(
        simple_graph, (5, 10, 0), 2, grow_to_lower_class=False
    )
    assert set(nb_edges) == {
        (1, 2, 0),
        (2, 1, 0),
        (6, 5, 0),
        (5, 6, 0),
        (6, 7, 0),
        (7, 6, 0),
        (10, 11, 0),
        (11, 10, 0),
        (11, 12, 0),
        (12, 11, 0),
        (10, 5, 0),
        (5, 10, 0),
        (2, 5, 0),
        (5, 2, 0),
    }
