import time
from ast import literal_eval
from collections import deque
from functools import lru_cache
from math import log

import geopandas as gpd
import momepy
import networkx as nx
import numpy as np
import osmnx
import pandas as pd
import rustworkx as rx
from networkx.algorithms.centrality.betweenness import (
    _accumulate_edges,
    _add_edge_keys,
    _rescale_e,
    _single_source_dijkstra_path_basic,
    _single_source_shortest_path_basic,
)
from networkx.utils import py_random_state
from shapely import LineString, Point, Polygon, unary_union
from shapely.ops import nearest_points
from tqdm import tqdm

from config import cfg

from .baseclass import GraphMetric, Metric


class RoadEdgeSpeeds(GraphMetric):
    attr_name = "speed_kph"
    priority = 2
    HWY_SPEEDS = {
        "motorway": 110,
        "trunk": 90,
        "primary": 80,
        "secondary": 70,
        "tertiary": 60,
        "unclassified": 50,
        "residential": 30,
        "motorway_link": 80,
        "trunk_link": 70,
        "primary_link": 60,
        "secondary_link": 50,
        "tertiary_link": 40,
        "living_street": 20,
        "service": 20,
        "pedestrian": 10,
        "track": 20,
        "road": 50,
    }

    def _calculate(self, road_graph: nx.MultiDiGraph, **kwargs) -> nx.MultiDiGraph:
        return osmnx.add_edge_speeds(
            road_graph, hwy_speeds=self.HWY_SPEEDS, fallback=50
        )


class RoadEdgeTravelTimes(GraphMetric):
    attr_name = "travel_time"
    priority = 3

    def _calculate(self, road_graph: nx.MultiDiGraph, **kwargs) -> nx.MultiDiGraph:
        return osmnx.add_edge_travel_times(road_graph)


# FIXME needs specific networkx version locked in, using private functions
@py_random_state(4)
# @nx._dispatch(edge_attrs="weight")
def _edge_betweenness_centrality(G, k=None, normalized=True, weight=None, seed=None):
    r"""Compute betweenness centrality for edges.

    Betweenness centrality of an edge $e$ is the sum of the
    fraction of all-pairs shortest paths that pass through $e$

    .. math::

       c_B(e) =\sum_{s,t \in V} \frac{\sigma(s, t|e)}{\sigma(s, t)}

    where $V$ is the set of nodes, $\sigma(s, t)$ is the number of
    shortest $(s, t)$-paths, and $\sigma(s, t|e)$ is the number of
    those paths passing through edge $e$ [2]_.

    Parameters
    ----------
    G : graph
      A NetworkX graph.

    k : int, optional (default=None)
      If k is not None use k node samples to estimate betweenness.
      The value of k <= n where n is the number of nodes in the graph.
      Higher values give better approximation.

    normalized : bool, optional
      If True the betweenness values are normalized by $2/(n(n-1))$
      for graphs, and $1/(n(n-1))$ for directed graphs where $n$
      is the number of nodes in G.

    weight : None or string, optional (default=None)
      If None, all edge weights are considered equal.
      Otherwise holds the name of the edge attribute used as weight.
      Weights are used to calculate weighted shortest paths, so they are
      interpreted as distances.

    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.
        Note that this is only used if k is not None.

    Returns
    -------
    edges : dictionary
       Dictionary of edges with betweenness centrality as the value.

    See Also
    --------
    betweenness_centrality
    edge_load

    Notes
    -----
    The algorithm is from Ulrik Brandes [1]_.

    For weighted graphs the edge weights must be greater than zero.
    Zero edge weights can produce an infinite number of equal length
    paths between pairs of nodes.

    References
    ----------
    .. [1]  A Faster Algorithm for Betweenness Centrality. Ulrik Brandes,
       Journal of Mathematical Sociology 25(2):163-177, 2001.
       https://doi.org/10.1080/0022250X.2001.9990249
    .. [2] Ulrik Brandes: On Variants of Shortest-Path Betweenness
       Centrality and their Generic Computation.
       Social Networks 30(2):136-145, 2008.
       https://doi.org/10.1016/j.socnet.2007.11.001
    """
    betweenness = dict.fromkeys(G, 0.0)  # b[v]=0 for v in G
    # b[e]=0 for e in G.edges()
    betweenness.update(dict.fromkeys(G.edges(), 0.0))
    if k is None:
        nodes = G
    else:
        nodes = seed.sample(list(G.nodes()), k)

    for s in tqdm(nodes, desc="Calculating edge betweenness"):
        # single source shortest paths
        if weight is None:  # use BFS
            S, P, sigma, _ = _single_source_shortest_path_basic(G, s)
        else:  # use Dijkstra's algorithm
            S, P, sigma, _ = _single_source_dijkstra_path_basic(G, s, weight)
        # accumulation
        betweenness = _accumulate_edges(betweenness, S, P, sigma, s)
    # rescaling
    for n in G:  # remove nodes to only return edges
        del betweenness[n]
    betweenness = _rescale_e(
        betweenness, len(G), normalized=normalized, directed=G.is_directed()
    )
    if G.is_multigraph():
        betweenness = _add_edge_keys(G, betweenness, weight=weight)
    return betweenness


def nx_to_rx(nonmulti_graph):
    weights = nx.get_edge_attributes(nonmulti_graph, "travel_time")
    # Creating the retworkx DiGraph
    rgraph = rx.PyDiGraph()
    node_map = {}
    # Ensuring all nodes are added first
    for node in nonmulti_graph.nodes():
        node_map[node] = rgraph.add_node(node)
    # Adding edges with the maximum travel_time
    for (u, v), travel_time in weights.items():
        rgraph.add_edge(node_map[u], node_map[v], travel_time)
    return rgraph, node_map


def build_osmid_map(graph):
    osmid_map = {}
    for u, v, data in graph.edges(data=True):
        osmids = data[cfg.layers.road_edges.unique_id]
        if isinstance(osmids, int):
            osmid_map[osmids] = (u, v)
        else:
            for osmid in osmids:
                osmid_map[osmid] = (u, v)

    return osmid_map


# class RoadEdgeBetweennessCentrality(GraphMetric):
#     attr_name = cfg.layers.road_edges.attributes.betweenness
#     BETWEENNESS_K_SAMPLE = 999
#
#     def _calculate(self, road_graph: nx.MultiDiGraph, **kwargs) -> nx.MultiDiGraph:
#         start_time = time.perf_counter()
#
#         simplified_graph = osmnx.simplify_graph(road_graph)
#
#         # has to be Graph not MultiGraph as MG is not supported by networkx2.4
#         graph = nx.Graph()
#         weight_column = "travel_time"
#         for u, v, key, data in simplified_graph.edges(data=True, keys=True):
#             if graph.has_edge(u, v):
#                 if (
#                     graph[u][v][weight_column]
#                     > simplified_graph[u][v][key][weight_column]
#                 ):
#                     nx.set_edge_attributes(graph, {(u, v): data})
#             else:
#                 graph.add_edge(u, v, **data)
#
#         total_node_count = len(graph.nodes())
#         k = (
#             self.BETWEENNESS_K_SAMPLE
#             if total_node_count > self.BETWEENNESS_K_SAMPLE
#             else None
#         )
#         if k:
#             print(
#                 f"Calculating betweenness centrality on subsample of nodes with k={k}, total node count={total_node_count}"
#             )
#         vals = _edge_betweenness_centrality(
#             graph, weight=weight_column, k=k, normalized=True
#         )
#         osmid_map = build_osmid_map(simplified_graph)
#         pb = tqdm(
#             total=len(road_graph.edges()), desc="Updating graph betweenness attribute."
#         )
#         for u, v, k, data in road_graph.edges(keys=True, data=True):
#             simplified_u, simplfied_v = osmid_map[data[cfg.layers.road_edges.unique_id]]
#             try:
#                 val = vals[simplified_u, simplfied_v]
#             except KeyError:
#                 val = vals[simplfied_v, simplified_u]
#             road_graph[u][v][k][self.attr_name] = val
#             pb.update(1)
#
#         pb.close()
#         print(
#             f"Betweenness centrality calculated in {time.perf_counter() - start_time:.2f} seconds"
#         )
#         return road_graph


class RoadEdgeDistanceToLastIntersection(GraphMetric):
    attr_name = cfg.layers.road_edges.attributes.distance_to_last_intersection

    def _calculate(self, road_graph: nx.MultiDiGraph, **kwargs) -> nx.MultiDiGraph:
        node_is_intersection = nx.get_node_attributes(
            road_graph, cfg.layers.road_nodes.attributes.is_intersection
        )

        def recursive_road_length_to_intersection(edge) -> float:
            u, v, key = edge
            length = road_graph.edges[edge]["length"]
            if node_is_intersection.get(u, True):  # True as default for node keyerror
                return length

            prev_v = u
            prev_u = next((n for n in road_graph.predecessors(u) if n != v), None)
            if not prev_u:
                return length
            return recursive_road_length_to_intersection((prev_u, prev_v, 0)) + length

        for edge in tqdm(road_graph.edges(keys=True)):
            length = recursive_road_length_to_intersection(edge)
            road_graph.edges[edge][self.attr_name] = length

        return road_graph


class RoadEdgeDistanceToLastMajorIntersection(GraphMetric):
    attr_name = cfg.layers.road_edges.attributes.distance_to_last_major_intersection

    def _calculate(self, road_graph: nx.MultiDiGraph, **kwargs) -> nx.MultiDiGraph:
        node_is_intersection = nx.get_node_attributes(
            road_graph, cfg.layers.road_nodes.attributes.is_major_intersection
        )

        def recursive_road_length_to_intersection(edge) -> float:
            u, v, key = edge
            length = road_graph.edges[edge]["length"]
            if node_is_intersection.get(u, True):  # True as default for node keyerror
                return length

            prev_v = u
            prev_u = next((n for n in road_graph.predecessors(u) if n != v), None)
            if not prev_u:
                return length
            return recursive_road_length_to_intersection((prev_u, prev_v, 0)) + length

        for *edge, type_category in tqdm(
            road_graph.edges(keys=True, data="type_category")
        ):
            if type_category not in ("major", "highway"):
                continue
            length = recursive_road_length_to_intersection(edge)
            road_graph.edges[edge][self.attr_name] = length

        return road_graph


class RoadEdgeNextNodeDegree(GraphMetric):
    """Not degree in the pure sense of the word. The complexities of a multidigraph make
    the pure degree not really usable for street generation in a non multi, non directional
    graph. This metric determines the amount of unique nodes that are neighbours of the end
    node of this street segment. This corresponds to:

    1: A dead end
    2: A continuing road section
    3: A T intersection
    4: A crossing intersection
    5...
    """

    attr_name = cfg.layers.road_edges.attributes.next_node_degree

    def _calculate(self, road_graph: nx.MultiDiGraph, **kwargs) -> nx.MultiDiGraph:
        for u, v, key in road_graph.edges(keys=True):
            predecessors = set(road_graph.predecessors(v))
            successors = set(road_graph.successors(v))

            # Combine predecessors and successors to get unique adjacent nodes
            unique_adjacent_nodes = predecessors.union(successors)

            road_graph[u][v][key][self.attr_name] = len(unique_adjacent_nodes)

        return road_graph


class RoadEdgeNextNodeMajorDegree(GraphMetric):
    """Not degree in the pure sense of the word. The complexities of a multidigraph make
    the pure degree not really usable for street generation in a non multi, non directional
    graph. This metric determines the amount of unique nodes that are neighbours of the end
    node of this street segment. This corresponds to:

    1: A dead end
    2: A continuing road section
    3: A T intersection
    4: A crossing intersection
    5...
    """

    attr_name = cfg.layers.road_edges.attributes.next_node_major_degree

    def _calculate(self, road_graph: nx.MultiDiGraph, **kwargs) -> nx.MultiDiGraph:
        node_degree = nx.get_node_attributes(road_graph, "node_major_degree")
        for u, v, key in road_graph.edges(keys=True):
            degree = node_degree.get(v)

            road_graph[u][v][key][self.attr_name] = degree

        return road_graph


class RoadEdgeForwardAngle(GraphMetric):
    attr_name = cfg.layers.road_edges.attributes.forward_angle
    priority = (
        2  # Calculate before RoadEdgeLeftAngle and RoadEdgePreviousSegmentForwardAngle
    )

    def _calculate(self, road_graph: nx.MultiDiGraph, **kwargs) -> nx.MultiDiGraph:
        x = nx.get_node_attributes(road_graph, "x")
        y = nx.get_node_attributes(road_graph, "y")
        forward_edges = {}
        forward_angles = {}
        forward_angles_abs = {}
        previous_edges = {}
        for u, v, key in road_graph.edges(keys=True):
            next_edges_vs = [node for node in road_graph.successors(v) if node != u]
            u_coords = x[u], y[u]
            v_coords = x[v], y[v]
            current_angle = np.arctan2(
                v_coords[1] - u_coords[1], v_coords[0] - u_coords[0]
            )
            next_angles = []
            next_edges = []
            for next_edge_v in next_edges_vs:
                next_coords = (x[next_edge_v], y[next_edge_v])
                next_angle = np.arctan2(
                    next_coords[1] - v_coords[1], next_coords[0] - v_coords[0]
                )
                angle_diff = np.degrees(next_angle - current_angle)
                # Normalize angle to be within [-180, 180]
                angle_diff = (angle_diff + 180) % 360 - 180
                next_angles.append(angle_diff)
                next_edges.append((v, next_edge_v, 0))
            if not next_angles:
                angle = None
                next_edge = None
            elif len(next_angles) == 1:
                angle = next_angles[0]
                next_edge = next_edges[0]
            else:
                angle = min(next_angles, key=lambda x: abs(x))
                next_edge = next_edges[next_angles.index(angle)]
                if abs(angle) > 45:
                    angle = None
                    next_edge = None

            forward_angles[(u, v, key)] = angle
            forward_angles_abs[(u, v, key)] = abs(angle) if angle is not None else None
            forward_edges[(u, v, key)] = next_edge
            if next_edge:
                previous_edges[next_edge] = (u, v, key)

        nx.set_edge_attributes(road_graph, forward_edges, "next_segment")
        nx.set_edge_attributes(road_graph, previous_edges, "previous_segment")
        nx.set_edge_attributes(road_graph, forward_angles, self.attr_name)
        nx.set_edge_attributes(road_graph, forward_angles_abs, self.attr_name + "_abs")

        return road_graph


class RoadEdgePreviousSegmentForwardAngle(Metric):
    attr_name = cfg.layers.road_edges.attributes.previous_segment_forward_angle

    def _calculate(self, road_edges: gpd.GeoDataFrame, **kwargs) -> pd.Series:
        previous_segment = road_edges["previous_segment"].dropna()
        if previous_segment.dtype == str:
            previous_segment = previous_segment.apply(literal_eval)

        forward_angle_map = pd.Series(
            road_edges[cfg.layers.road_edges.attributes.forward_angle].values,
            index=road_edges.index,
        )

        previous_forward_angles = forward_angle_map.loc[previous_segment].values

        result = pd.Series(index=road_edges.index, dtype=float)
        result.loc[previous_segment.index] = previous_forward_angles

        return result


#
# class RoadEdgePreviousSegmentForwardAngle(GraphMetric):
#     attr_name = cfg.layers.road_edges.attributes.previous_segment_forward_angle
#
#     def _calculate(self, road_graph: nx.MultiDiGraph, **kwargs) -> nx.MultiDiGraph:
#
#         for edge in road_graph.edges(keys=True):
#             previous_edge = road_graph.edges[edge].get("previous_segment")
#             if not previous_edge:
#                 continue
#             if isinstance(previous_edge, str):
#                 previous_edge = literal_eval(previous_edge)
#
#             road_graph.edges[edge][self.attr_name] = road_graph.edges[previous_edge][
#                 cfg.layers.road_edges.attributes.forward_angle
#             ]
#
#         return road_graph


class RoadSectionID(GraphMetric):
    attr_name = cfg.layers.road_edges.attributes.section_id
    priority = 5  # After intersection but before others

    def _calculate(self, road_graph: nx.MultiDiGraph, **kwargs) -> nx.MultiDiGraph:
        visited = set()
        section_id = 0
        node_degree_map = nx.get_node_attributes(
            road_graph, cfg.layers.road_nodes.attributes.node_degree
        )
        for edge in tqdm(road_graph.edges(keys=True)):
            if edge in visited:
                continue
            section = get_road_section(edge, road_graph, node_degree_map)
            for e in section:
                road_graph.edges[e][self.attr_name] = section_id
                visited.add(e)
            section_id += 1

        return road_graph


class RoadMajorSectionID(GraphMetric):
    attr_name = cfg.layers.road_edges.attributes.major_section_id
    priority = 5  # After intersection but before others

    def _calculate(self, road_graph: nx.MultiDiGraph, **kwargs) -> nx.MultiDiGraph:
        visited = set()
        section_id = 0
        node_degree_map = nx.get_node_attributes(
            road_graph, cfg.layers.road_nodes.attributes.node_major_degree
        )
        type_category = nx.get_edge_attributes(road_graph, "type_category")
        is_major = {
            e: type_cat in ("highway", "major") for e, type_cat in type_category.items()
        }
        major_section_id = {}
        for edge in tqdm(road_graph.edges(keys=True)):
            if not is_major.get(edge, False):
                continue
            if edge in visited:
                continue
            section = get_major_road_section(edge, road_graph, node_degree_map)
            for e in section:
                major_section_id[e] = section_id
                visited.add(e)
            section_id += 1

        nx.set_edge_attributes(road_graph, major_section_id, self.attr_name)
        return road_graph


class RoadEdgeOnDomainBorder(Metric):
    attr_name = cfg.layers.road_edges.attributes.on_domain_border

    def _calculate(
        self, road_edges: gpd.GeoDataFrame, extents: gpd.GeoDataFrame, **kwargs
    ) -> gpd.GeoDataFrame:
        extents_poly = unary_union(extents.geometry)
        # FIXME, correct this if extents is no longer bounding box of admin area
        extents_poly = Polygon.from_bounds(*extents_poly.bounds)

        series = road_edges.geometry.apply(
            lambda geom: geom.distance(extents_poly.boundary) < cfg.domain_border_buffer
        )

        return series


def angle_shannon_entropy(data: np.array, bins=36, offset=5):
    hist, bin_edges = np.histogram(data + offset, bins=bins, range=(-180, 180))
    hist = hist / hist.sum()
    hist = hist[hist > 0]
    return -np.sum(hist * np.log2(hist))


class RoadSectionEndsInDeadEnd(Metric):
    attr_name = cfg.layers.road_edges.attributes.dead_end_section

    def _calculate(
        self,
        road_nodes: gpd.GeoDataFrame,
        road_edges: gpd.GeoDataFrame,
        road_graph: nx.MultiDiGraph,
        **kwargs,
    ) -> gpd.GeoDataFrame:
        # group by section id, see if any node in the section is a dead end
        road_edges[self.attr_name] = False
        road_edges.sort_index(inplace=True)

        dead_end_nodes = set(road_nodes[road_nodes["node_degree"] == 1].index)

        not_on_domain_border = ~road_edges[
            cfg.layers.road_edges.attributes.on_domain_border
        ]

        # Create a mask for rows where either node is a dead end
        is_dead_end = road_edges.index.get_level_values(0).isin(
            dead_end_nodes
        ) | road_edges.index.get_level_values(1).isin(dead_end_nodes)

        # Combine the masks
        final_mask = not_on_domain_border & is_dead_end

        # Set the attribute where the final mask is True
        road_edges.loc[final_mask, self.attr_name] = True

        return road_edges.groupby(cfg.layers.road_edges.attributes.section_id)[
            self.attr_name
        ].transform("any")


class RoadEdgePreviousSegmentLength(Metric):
    attr_name = cfg.layers.road_edges.attributes.previous_segment_length

    def _calculate(self, road_edges: gpd.GeoDataFrame, **kwargs) -> pd.Series:
        previous_segment = road_edges["previous_segment"].dropna()
        if previous_segment.dtype == str:
            previous_segment = previous_segment.apply(literal_eval)

        forward_angle_map = pd.Series(
            road_edges["length"].values,
            index=road_edges.index,
        )

        previous_values = forward_angle_map.loc[previous_segment].values

        result = pd.Series(index=road_edges.index, dtype=float)
        result.loc[previous_segment.index] = previous_values

        return result


def get_previous_edge(edge, road_graph):
    previous_edge = road_graph.edges[edge].get("previous_segment")
    if isinstance(previous_edge, str):
        previous_edge = literal_eval(previous_edge)
    return previous_edge


def get_next_edge(edge, road_graph):
    next_edge = road_graph.edges[edge].get("next_segment")
    if isinstance(next_edge, str):
        next_edge = literal_eval(next_edge)
    return next_edge


class RoadEdgeBearing(GraphMetric):
    attr_name = cfg.layers.road_edges.attributes.bearing

    def _calculate(self, road_graph: nx.MultiDiGraph, **kwargs) -> nx.MultiDiGraph:
        for u, v, key in road_graph.edges(keys=True):
            u_coords = road_graph.nodes[u]["x"], road_graph.nodes[u]["y"]
            v_coords = road_graph.nodes[v]["x"], road_graph.nodes[v]["y"]
            bearing = np.arctan2(v_coords[1] - u_coords[1], v_coords[0] - u_coords[0])
            road_graph[u][v][key][self.attr_name] = np.degrees(bearing)

        return road_graph


class RoadEdgeContinuity(GraphMetric):
    attr_name = cfg.layers.road_edges.attributes.continuity

    def _calculate(self, road_graph: nx.MultiDiGraph, **kwargs) -> nx.MultiDiGraph:
        MAX_LENGTH = 1000
        ANGLE_THRESHOLD = 25

        for edge in tqdm(road_graph.edges(keys=True)):
            queue = deque([edge])
            continuity_length = 0
            visited = set()
            while queue:
                next_edge = queue.popleft()
                visited.add(next_edge)
                continuity_length += road_graph.edges[next_edge]["length"]
                if continuity_length >= MAX_LENGTH:
                    continuity_length = MAX_LENGTH
                    break
                previous_segment = get_previous_edge(next_edge, road_graph)
                if previous_segment and previous_segment not in visited:
                    angle_with_current_segment = road_graph.edges[previous_segment][
                        "forward_angle"
                    ]
                    if abs(angle_with_current_segment) < ANGLE_THRESHOLD:
                        queue.append(previous_segment)
                next_segment = get_next_edge(next_edge, road_graph)
                if next_segment and next_segment not in visited:
                    angle_with_current_segment = road_graph.edges[next_edge][
                        "forward_angle"
                    ]
                    if abs(angle_with_current_segment) < ANGLE_THRESHOLD:
                        queue.append(next_segment)

            road_graph.edges[edge][self.attr_name] = continuity_length / MAX_LENGTH

        return road_graph


# class RoadEdgeCurviness(GraphMetric):
#     attr_name = cfg.layers.road_edges.attributes.curviness
#
#     def _calculate(self, road_graph: nx.MultiDiGraph, **kwargs) -> nx.MultiDiGraph:
#         for u, v, key in tqdm(road_graph.edges(keys=True)):
#             forward_angle = road_graph[u][v][key].get(
#                 cfg.layers.road_edges.attributes.forward_angle
#             )
#             previous_edge = get_previous_edge((u, v, key), road_graph)
#             if not previous_edge or not forward_angle:
#                 continue
#             previous_angle = road_graph.edges[previous_edge].get(
#                 cfg.layers.road_edges.attributes.forward_angle
#             )
#             length = road_graph.edges[u, v, key]["length"]
#
#             curvature = abs(forward_angle - previous_angle) / length
#
#             road_graph[u][v][key][self.attr_name] = curvature
#
#         return road_graph


EdgeIndex = tuple[float, float, float]


def _get_road_section(edge, graph):
    # Cache based on edge and the number of nodes in the graph
    @lru_cache(None)
    def cached_get_road_section(edge, graph_length):
        return _get_road_section(edge, graph)

    # Call the cached function with the length of the graph as part of the key
    return cached_get_road_section(edge, len(graph))


def get_road_section(edge, graph, node_degree: dict[int, int]) -> list[EdgeIndex]:
    u, v, key = edge
    section = deque([(u, v, 0)])

    # grow both directions until is_intersection
    while node_degree[u] == 2:
        prev_v = u
        prev_u = next((n for n in graph.predecessors(u) if n != v), None)
        if not prev_u:
            break
        section.appendleft((prev_u, prev_v, 0))
        u = prev_u
        v = prev_v

    u, v, key = edge
    while node_degree[v] == 2:
        next_u = v
        next_v = next((n for n in graph.successors(v) if n != u), None)
        if not next_v:
            break
        section.append((next_u, next_v, 0))
        u = next_u
        v = next_v

    return list(section)


def get_major_road_section(
    edge, graph: nx.MultiDiGraph, node_degree: dict[int, int]
) -> list[EdgeIndex]:
    u, v, key = edge
    section = deque([(u, v, 0)])
    visited = set()

    # grow both directions until is_intersection
    while node_degree[u] == 2:
        previous_edge = graph.get_edge_data(u, v, 0).get("previous_segment")
        if isinstance(previous_edge, str):
            previous_edge = literal_eval(previous_edge)
        if not previous_edge:
            break
        if previous_edge in visited:
            break
        prev_u, prev_v, _ = previous_edge
        if graph.get_edge_data(prev_u, prev_v, 0).get("type_category") not in (
            "major",
            "highway",
        ):
            break
        section.appendleft((prev_u, prev_v, 0))
        visited.add(previous_edge)
        u = prev_u
        v = prev_v

    u, v, key = edge
    while node_degree[v] == 2:
        next_edge = graph.get_edge_data(u, v, 0).get("next_segment")
        if isinstance(next_edge, str):
            next_edge = literal_eval(next_edge)
        if not next_edge:
            break
        if next_edge in visited:
            break
        next_u, next_v, _ = next_edge
        if graph.get_edge_data(next_u, next_v, 0).get("type_category") not in (
            "major",
            "highway",
        ):
            break
        section.append((next_u, next_v, 0))
        visited.add(next_edge)
        u = next_u
        v = next_v

    return list(section)


class RoadEdgeSectionLength(Metric):
    attr_name = cfg.layers.road_edges.attributes.section_length

    def _calculate(self, road_edges: gpd.GeoDataFrame, **kwargs) -> nx.MultiDiGraph:
        return road_edges.groupby(cfg.layers.road_edges.attributes.section_id)[
            "length"
        ].transform("sum")


class RoadEdgeMajorSectionLength(Metric):
    attr_name = cfg.layers.road_edges.attributes.major_section_length

    def _calculate(self, road_edges: gpd.GeoDataFrame, **kwargs) -> nx.MultiDiGraph:
        return (
            road_edges[road_edges["type_category"].isin(("major", "highway"))]
            .groupby(cfg.layers.road_edges.attributes.major_section_id)["length"]
            .transform("sum")
        )


class RoadEdgeStretchLinearity(GraphMetric):
    attr_name = cfg.layers.road_edges.attributes.stretch_linearity
    priority = 4  # Before RoadEdgeStretchCurvilinearity

    def _calculate(self, road_graph: nx.MultiDiGraph, **kwargs) -> nx.MultiDiGraph:
        for edge in tqdm(road_graph.edges(keys=True)):
            stretch = get_stretch(edge, road_graph, max_angle=40)

            start = (
                road_graph.nodes[stretch[0][0]]["x"],
                road_graph.nodes[stretch[0][0]]["y"],
            )
            end = (
                road_graph.nodes[stretch[-1][1]]["x"],
                road_graph.nodes[stretch[-1][1]]["y"],
            )
            # from http://docs.momepy.org/en/stable/generated/momepy.Linearity.html
            # also called Sinuosity https://gis.stackexchange.com/questions/469882/is-there-a-common-method-for-measuring-curviness-of-a-line
            l_euclidean = np.linalg.norm(np.array(start) - np.array(end))
            l_segment = sum(road_graph.edges[e]["length"] for e in stretch)
            section_linearity = l_euclidean / l_segment

            road_graph.edges[edge][self.attr_name] = log(
                1 / (1 - min(section_linearity, 0.9999))
            ) / log(1 / (1 - 0.9999))

        return road_graph


class RoadEdgeStretchCurvilinearity(Metric):
    attr_name = cfg.layers.road_edges.attributes.stretch_curvilinearity

    def _calculate(self, road_edges: gpd.GeoDataFrame, **kwargs) -> nx.MultiDiGraph:
        return 1 - road_edges[RoadEdgeStretchLinearity.attr_name]


def get_stretch(edge, road_graph, max_length=500, max_angle=25):
    queue = deque([(edge, "both")])
    full_stretch = deque([edge])
    visited = set()
    current_length = 0
    while queue:
        next_edge, direction = queue.popleft()
        visited.add(next_edge)
        if direction == "backwards":
            full_stretch.appendleft(next_edge)
        elif direction == "forwards":
            full_stretch.append(next_edge)

        current_length += road_graph.edges[next_edge]["length"]
        if current_length >= max_length:
            break
        if direction in ("both", "backwards"):
            previous_segment = get_previous_edge(next_edge, road_graph)
            if previous_segment and previous_segment not in visited:
                angle_with_current_segment = road_graph.edges[previous_segment][
                    "forward_angle"
                ]
                if abs(angle_with_current_segment) < max_angle:
                    queue.append((previous_segment, "backwards"))
        if direction in ("both", "forwards"):
            next_segment = get_next_edge(next_edge, road_graph)
            if next_segment and next_segment not in visited:
                angle_with_current_segment = road_graph.edges[next_edge][
                    "forward_angle"
                ]
                if abs(angle_with_current_segment) < max_angle:
                    queue.append((next_segment, "forwards"))
    return list(full_stretch)


# class RoadEdgeSectionCumulativeAngle(GraphMetric):
#     attr_name = cfg.layers.road_edges.attributes.section_cumulative_angle
#
#     def _calculate(self, road_graph: nx.MultiDiGraph, **kwargs) -> nx.MultiDiGraph:
#         visited = set()
#         for edge in tqdm(road_graph.edges(keys=True)):
#             if edge in visited:
#                 continue
#             section = get_road_section(edge, road_graph)
#
#             cumulative_angle = 0
#             for e in section:
#                 forward_angle = road_graph.edges[e].get(
#                     cfg.layers.road_edges.attributes.forward_angle
#                 )
#                 if not forward_angle:
#                     continue
#                 # length = road_graph.edges[e]["length"]
#                 cumulative_angle += abs(forward_angle)
#
#             # total_length = sum(road_graph.edges[e]["length"] for e in section)
#             # curvature /= total_length
#
#             for e in section:
#                 road_graph.edges[e][self.attr_name] = cumulative_angle
#                 visited.add(e)
#
#         return road_graph
#


class RoadEdgeIntersectionLeftAngle(GraphMetric):
    attr_name = cfg.layers.road_edges.attributes.intersection_left_angle

    def _calculate(self, road_graph: nx.MultiDiGraph, **kwargs) -> nx.MultiDiGraph:
        is_intersection = nx.get_node_attributes(
            road_graph, cfg.layers.road_nodes.attributes.is_intersection
        )
        x = nx.get_node_attributes(road_graph, "x")
        y = nx.get_node_attributes(road_graph, "y")
        forward_angles = nx.get_edge_attributes(
            road_graph, cfg.layers.road_edges.attributes.forward_angle
        )
        left_angles = {}
        for u, v, key in road_graph.edges(keys=True):
            if not is_intersection.get(v):
                continue

            next_edges_vs = [node for node in road_graph.successors(v) if node != u]
            u_coords = x[u], y[u]
            v_coords = x[v], y[v]
            current_angle = np.arctan2(
                v_coords[1] - u_coords[1], v_coords[0] - u_coords[0]
            )

            next_angles = []
            for next_edge_v in next_edges_vs:
                next_coords = (
                    x[next_edge_v],
                    y[next_edge_v],
                )
                next_angle = np.arctan2(
                    next_coords[1] - v_coords[1], next_coords[0] - v_coords[0]
                )
                angle_diff = np.degrees(next_angle - current_angle)
                # Normalize angle to be within [-180, 180]
                angle_diff = (angle_diff + 180) % 360 - 180
                next_angles.append(angle_diff)

            if not next_angles:
                left_angles[(u, v, key)] = None
                continue

            left_angle = max(next_angles)
            forward_angle = forward_angles.get((u, v, key))
            # If the angle is negative, the angle is to the right. There is no left segment if the
            # angle is to the right or the angle is the same as the forward angle, so skip
            if left_angle < 0 or left_angle == forward_angle:
                left_angle = None

            left_angles[(u, v, key)] = left_angle

        nx.set_edge_attributes(road_graph, left_angles, self.attr_name)
        return road_graph


class RoadEdgeMajorIntersectionLeftAngle(GraphMetric):
    attr_name = cfg.layers.road_edges.attributes.major_intersection_left_angle

    def _calculate(self, road_graph: nx.MultiDiGraph, **kwargs) -> nx.MultiDiGraph:
        is_major_intersection = nx.get_node_attributes(
            road_graph, cfg.layers.road_nodes.attributes.is_major_intersection
        )
        for u, v, key in road_graph.edges(keys=True):
            if not is_major_intersection.get(v):
                continue
            if road_graph.get_edge_data(u, v).get("type_category") not in (
                "major",
                "highway",
            ):
                continue
            next_edges_vs = [node for node in road_graph.successors(v) if node != u]
            u_coords = road_graph.nodes[u]["x"], road_graph.nodes[u]["y"]
            v_coords = road_graph.nodes[v]["x"], road_graph.nodes[v]["y"]
            current_angle = np.arctan2(
                v_coords[1] - u_coords[1], v_coords[0] - u_coords[0]
            )

            next_angles = []
            for next_edge_v in next_edges_vs:
                next_coords = (
                    road_graph.nodes[next_edge_v]["x"],
                    road_graph.nodes[next_edge_v]["y"],
                )
                next_angle = np.arctan2(
                    next_coords[1] - v_coords[1], next_coords[0] - v_coords[0]
                )
                angle_diff = np.degrees(next_angle - current_angle)
                # Normalize angle to be within [-180, 180]
                angle_diff = (angle_diff + 180) % 360 - 180
                next_angles.append(angle_diff)

            if not next_angles:
                road_graph[u][v][key][self.attr_name] = None
                continue

            left_angle = max(next_angles)
            forward_angle = road_graph[u][v][key].get(
                cfg.layers.road_edges.attributes.forward_angle
            )
            # If the angle is negative, the angle is to the right. There is no left segment if the
            # angle is to the right or the angle is the same as the forward angle, so skip
            if left_angle < 0 or left_angle == forward_angle:
                left_angle = None

            road_graph[u][v][key][self.attr_name] = left_angle

        return road_graph


# FIXME KeyError spatial weights
# class RoadEdgeSegmentLength(Metric):
#     attr_name = "segment_length"
#
#     def calculate(self, road_edges: gpd.GeoDataFrame, **kwargs) -> gpd.GeoDataFrame:
#         return momepy.SegmentsLength(road_edges).series


# class RoadEdgeLinearity(Metric):
#     attr_name = cfg.layers.road_edges.attributes.linearity
#
#     def _calculate(self, road_edges: gpd.GeoDataFrame, **kwargs) -> gpd.GeoDataFrame:
#         return momepy.Linearity(road_edges).series


class RoadEdgeNeighboringStreetOrientationDeviation(Metric):
    attr_name = (
        cfg.layers.road_edges.attributes.neighboring_street_orientation_deviation
    )

    def _calculate(self, road_edges: gpd.GeoDataFrame, **kwargs) -> gpd.GeoDataFrame:
        return momepy.NeighboringStreetOrientationDeviation(road_edges).series


# class RoadsCoinsLength(Metric):
#     attr_name = cfg.layers.road_edges.attributes.coins_length
#
#     def _calculate(self, road_edges: gpd.GeoDataFrame, **kwargs) -> gpd.GeoDataFrame:
#         coins = momepy.COINS(road_edges)
#         road_edges["COINS_group"] = coins.stroke_attribute()
#         coins_edges = coins.stroke_gdf()
#
#         coins_edges["length"] = coins_edges.geometry.length
#
#         length_dict = coins_edges["length"].to_dict()
#
#         return road_edges["COINS_group"].map(length_dict)


# class RoadEnclosureIDs(Metric):
#     attr_name = cfg.layers.road_edges.attributes.enclosure_ids
#
#     def _calculate(
#         self, road_edges: gpd.GeoDataFrame, enclosures: gpd.GeoDataFrame, **kwargs
#     ) -> gpd.GeoDataFrame:
#         touching_enclosures = gpd.sjoin(
#             road_edges, enclosures[enclosures.is_valid], predicate="touches"
#         )
#         grouped = touching_enclosures.groupby(touching_enclosures.index)
#         result = grouped.apply(lambda x: x[cfg.layers.enclosures.unique_id].tolist())
#         # result = result.reindex(road_edges.index, fill_value=None)
#         return result
#


class RoadEdgeCityCenterBearing(Metric):
    attr_name = cfg.layers.road_edges.attributes.city_center_bearing

    def _calculate(
        self, road_edges: gpd.GeoDataFrame, city_center: gpd.GeoDataFrame, **kwargs
    ) -> gpd.GeoDataFrame:
        city_center = Point(city_center.geometry.x[0], city_center.geometry.y[0])

        def bearing_to_city_center(edge):
            edge_vector = np.array(
                [
                    edge.coords[0][0] - edge.coords[-1][0],
                    edge.coords[0][1] - edge.coords[-1][1],
                ]
            )
            city_center_vector = np.array(
                [city_center.x - edge.coords[-1][0], city_center.y - edge.coords[-1][1]]
            )
            angle = np.arccos(
                np.dot(edge_vector, city_center_vector)
                / (np.linalg.norm(edge_vector) * np.linalg.norm(city_center_vector))
            )
            angle = np.degrees(angle)
            normalized_angle = (180 - angle) % 180
            if normalized_angle > 90:
                normalized_angle = 180 - normalized_angle
            return normalized_angle

        return road_edges.geometry.apply(lambda x: bearing_to_city_center(x))


class RoadEdgeRightNeighbour(Metric):
    attr_name = cfg.layers.road_edges.attributes.right_neighbour
    priority = 4  # before RoadEdgeRightNeighbourAngleDeviation and RoadEdgeRightNeighbourDistance

    def _calculate(self, road_edges: gpd.GeoDataFrame, **kwargs) -> pd.Series:
        right_neigbour = {}
        threshold_distance = 6
        assert road_edges.index.names == ["u", "v", "key"]

        centroids = road_edges.geometry.centroid
        centroids_x = centroids.x.to_numpy()
        centroids_y = centroids.y.to_numpy()
        centroids_np = np.column_stack((centroids_x, centroids_y))
        start_points = np.array([list(x.coords[0]) for x in road_edges.geometry])
        end_points = np.array([list(x.coords[-1]) for x in road_edges.geometry])

        dx = end_points[:, 0] - start_points[:, 0]
        dy = end_points[:, 1] - start_points[:, 1]
        ortho_lines = [
            LineString([(x, y), (x - dy, y + dx)])
            for (x, y), dx, dy in zip(centroids_np, dx, dy)
        ]
        max_distance = cfg.layers.road_edges.attributes.right_neighbour_max_distance_m
        ortho_lines_scaled = [
            self._scale_line(line, max_distance) for line in ortho_lines
        ]
        ortho_lines_scaled = gpd.GeoSeries(ortho_lines_scaled)

        # Define the chunk size
        chunk_size = 1000
        pbar = tqdm(total=len(road_edges))

        # Iterate over DataFrame in chunks
        for start in range(0, len(road_edges), chunk_size):
            road_edge_chunk = road_edges.iloc[start : start + chunk_size]
            road_edge_ortho = ortho_lines_scaled.iloc[
                start : start + chunk_size
            ].to_frame()
            road_edge_ortho.crs = road_edges.crs
            road_edge_ortho["original_iloc"] = road_edge_ortho.index.to_numpy() - start
            # Convert road_edge_ortho to GeoDataFrame if it's not already
            if isinstance(road_edge_ortho, gpd.GeoSeries):
                road_edge_ortho = gpd.GeoDataFrame(
                    geometry=road_edge_ortho, crs=road_edges.crs
                )

            ortho_intersections = gpd.sjoin(
                road_edge_ortho, road_edges, predicate="crosses", how="inner"
            )

            ortho_intersections = ortho_intersections.set_index(["u", "v", "key"])
            target_edges = road_edges.join(
                ortho_intersections,
                on=("u", "v", "key"),
                how="inner",
                rsuffix="_right",
            )
            t_e = target_edges.reset_index()
            source_centroids = road_edge_chunk.iloc[
                t_e["original_iloc"].to_numpy()
            ].geometry.centroid
            distances = t_e.geometry.distance(source_centroids.reset_index())
            valid_distances = distances[distances >= threshold_distance].to_frame()
            valid_distances["original_iloc"] = t_e["original_iloc"]
            valid_distances["target_u"] = t_e["u"]
            valid_distances["target_v"] = t_e["v"]
            valid_distances["target_key"] = t_e["key"]

            for original_iloc, distance in valid_distances.groupby("original_iloc"):
                uvk = road_edge_chunk.iloc[original_iloc].name
                min_dist_idx = distance.idxmin().loc[0]
                min_dist_uvk = valid_distances.loc[
                    min_dist_idx, ["target_u", "target_v", "target_key"]
                ]

                right_neigbour[uvk] = tuple(map(int, min_dist_uvk))

            pbar.update(chunk_size)

        right_nb_series = pd.Series(
            right_neigbour.values(),
            index=right_neigbour.keys(),
        )

        return right_nb_series

    def _scale_line(self, line, length):
        """Scale the line to a specific length, starting 1 meter away from the source point."""
        p1, p2 = line.coords[:]
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        original_length = np.sqrt(dx**2 + dy**2)

        # Adjust the starting point to be 1 meter away from p1 towards p2
        if original_length > 1:
            scale_start = 1 / original_length
            p1_new = (p1[0] + dx * scale_start, p1[1] + dy * scale_start)
        else:
            p1_new = p1  # If the original line is less than 1 meter, keep p1 as is

        # Now scale the line to the desired length from the new starting point
        scale = length / original_length
        new_line = LineString(
            [p1_new, (p1_new[0] + dx * scale, p1_new[1] + dy * scale)]
        )

        return new_line


class RoadEdgeMajorRightNeighbour(Metric):
    attr_name = cfg.layers.road_edges.attributes.right_major_neighbour
    priority = 4  # before RoadEdgeRightNeighbourAngleDeviation and RoadEdgeRightNeighbourDistance

    def _calculate(self, road_edges: gpd.GeoDataFrame, **kwargs) -> pd.Series:
        right_neigbour = {}
        threshold_distance = 50
        road_edges_major = road_edges[
            road_edges["type_category"].isin(["major", "highway"])
        ]
        assert road_edges_major.index.names == ["u", "v", "key"]

        for idx, road_edge in tqdm(road_edges_major.iterrows()):
            # Compute the centroid of the road_edge
            centroid = road_edge.geometry.centroid
            # Calculate the orthogonal direction
            dx = road_edge.geometry.coords[-1][0] - road_edge.geometry.coords[0][0]
            dy = road_edge.geometry.coords[-1][1] - road_edge.geometry.coords[0][1]
            ortho_line = LineString(
                [(centroid.x, centroid.y), (centroid.x - dy, centroid.y + dx)]
            )

            # Scale orthogonal line to a specific length
            ortho_line = self._scale_line(
                ortho_line,
                cfg.layers.road_edges.attributes.right_major_neighbour_max_distance_m,
            )

            intersections = road_edges_major.sindex.query(
                ortho_line, predicate="crosses"
            )
            # Find the closest intersection with this line
            closest_road, min_distance = None, float("inf")

            for intersection in intersections:
                other_edge = road_edges_major.iloc[intersection]
                distance = centroid.distance(other_edge.geometry)
                if min_distance > distance > threshold_distance:
                    closest_road = other_edge.name
                    min_distance = distance

            right_neigbour[idx] = closest_road

        right_nb_series = pd.Series(
            right_neigbour.values(),
            index=right_neigbour.keys(),
        )

        return right_nb_series

    def _scale_line(self, line, length):
        """Scale the line to a specific length."""
        p1, p2 = line.coords[:]
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        scale = length / np.sqrt(dx**2 + dy**2)
        new_line = LineString(
            [(p1[0], p1[1]), (p1[0] + dx * scale, p1[1] + dy * scale)]
        )
        return new_line


class RoadEdgeRightNeighbourAngleDeviation(Metric):
    attr_name = cfg.layers.road_edges.attributes.right_neighbour_angle_deviation

    def _calculate(self, road_edges: gpd.GeoDataFrame, **kwargs) -> pd.Series:
        results = []
        assert road_edges.index.names == ["u", "v", "key"]

        for idx, road_edge in tqdm(road_edges.iterrows()):
            right_neigbour = road_edge[RoadEdgeRightNeighbour.attr_name]
            if right_neigbour is None or right_neigbour == "None":  # No right neighbour
                results.append((idx, None))
                continue

            # If calculated in a prevuous run it was turned into a string for gpkg saving
            if isinstance(right_neigbour, str):
                right_neigbour = literal_eval(right_neigbour)
            if pd.isna(right_neigbour) or pd.isna(right_neigbour[0]):
                continue

            angle_deviation = _calculate_angle_deviation(
                road_edge.geometry, road_edges.loc[right_neigbour].geometry
            )
            results.append((idx, angle_deviation))

        angle_deviation_series = pd.Series(
            [angle_deviation for _, angle_deviation in results],
            index=[idx for idx, _ in results],
        )

        return angle_deviation_series


class RoadEdgeRightMajorNeighbourAngleDeviation(Metric):
    attr_name = cfg.layers.road_edges.attributes.right_major_neighbour_angle_deviation

    def _calculate(self, road_edges: gpd.GeoDataFrame, **kwargs) -> pd.Series:
        results = []
        assert road_edges.index.names == ["u", "v", "key"]
        road_edges_major = road_edges[
            road_edges["type_category"].isin(["major", "highway"])
        ]
        for idx, road_edge in tqdm(road_edges_major.iterrows()):
            right_neigbour = road_edge[RoadEdgeMajorRightNeighbour.attr_name]
            if right_neigbour is None or right_neigbour == "None":  # No right neighbour
                results.append((idx, None))
                continue

            # If calculated in a prevuous run it was turned into a string for gpkg saving
            if isinstance(right_neigbour, str):
                right_neigbour = literal_eval(right_neigbour)

            angle_deviation = _calculate_angle_deviation(
                road_edge.geometry, road_edges_major.loc[right_neigbour].geometry
            )
            results.append((idx, angle_deviation))

        angle_deviation_series = pd.Series(
            [angle_deviation for _, angle_deviation in results],
            index=[idx for idx, _ in results],
        )

        return angle_deviation_series


def angle(line):
    x_diff = line.coords[1][0] - line.coords[0][0]
    y_diff = line.coords[1][1] - line.coords[0][1]
    return np.arctan2(y_diff, x_diff)


def _calculate_angle_deviation(geometry1, geometry2):
    """Calculate the angle deviation between two lines, where the maximum deviation is 90 degrees."""

    angle1 = angle(geometry1)
    angle2 = angle(geometry2)
    angle_diff = (
        np.degrees(angle1 - angle2) % 360
    )  # Calculate the absolute difference in degrees

    # Normalize the difference to be within 0 to 180 degrees, where 180 is treated as 0
    if angle_diff > 180:
        angle_diff = 360 - angle_diff

    # Map the difference to be within 0 to 90 degrees
    if angle_diff > 90:
        angle_diff = 180 - angle_diff

    return angle_diff


class RoadEdgeRightNeighbourDistance(Metric):
    attr_name = cfg.layers.road_edges.attributes.right_neighbour_distance

    def _calculate(self, road_edges: gpd.GeoDataFrame, **kwargs) -> pd.Series:
        results = []
        assert road_edges.index.names == ["u", "v", "key"]

        for idx, road_edge in tqdm(road_edges.iterrows()):
            right_neigbour = road_edge[RoadEdgeRightNeighbour.attr_name]
            # No right neighbour
            if right_neigbour in (None, "None", "NaN", "nan", np.nan):
                results.append((idx, None))
                continue

            # If calculated in a prevuous run it was turned into a string for gpkg saving
            if isinstance(right_neigbour, str):
                right_neigbour = literal_eval(right_neigbour)

            distance = road_edge.geometry.distance(
                road_edges.loc[right_neigbour].geometry
            )

            results.append((idx, distance))

        result = pd.Series(
            [distance for _, distance in results],
            index=[idx for idx, _ in results],
        )

        return result


class RoadEdgeRightMajorNeighbourDistance(Metric):
    attr_name = cfg.layers.road_edges.attributes.right_major_neighbour_distance

    def _calculate(self, road_edges: gpd.GeoDataFrame, **kwargs) -> pd.Series:
        results = []
        assert road_edges.index.names == ["u", "v", "key"]
        road_major_edges = road_edges[
            road_edges["type_category"].isin(["major", "highway"])
        ]

        for idx, road_edge in tqdm(road_major_edges.iterrows()):
            right_neigbour = road_edge[RoadEdgeMajorRightNeighbour.attr_name]
            # No right neighbour
            if right_neigbour in (None, "None", "NaN", "nan", np.nan):
                results.append((idx, None))
                continue

            # If calculated in a prevuous run it was turned into a string for gpkg saving
            if isinstance(right_neigbour, str):
                right_neigbour = literal_eval(right_neigbour)

            distance = road_edge.geometry.distance(
                road_major_edges.loc[right_neigbour].geometry
            )

            results.append((idx, distance))

        result = pd.Series(
            [distance for _, distance in results],
            index=[idx for idx, _ in results],
        )

        return result
