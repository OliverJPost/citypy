import geopandas as gpd
import momepy
import networkx as nx
import pandas as pd

from config import cfg

from ...road_class import EdgeClass
from .baseclass import GraphMetric, Metric


def add_node_data(graph: nx.Graph, data_dict: dict, attr_name: str) -> None:
    for node in graph.nodes(data=True):
        node[1][attr_name] = data_dict[node[0]]


class RoadNodeIsIntersection(GraphMetric):
    attr_name = cfg.layers.road_nodes.attributes.is_intersection
    priority = 4  # before RoadEdgeDistanceToLastIntersection

    def _calculate(self, road_graph: nx.MultiDiGraph, **kwargs) -> nx.MultiDiGraph:
        node_degree = nx.get_node_attributes(
            road_graph, cfg.layers.road_nodes.attributes.node_degree
        )
        is_intersection = {node: degree > 2 for node, degree in node_degree.items()}
        nx.set_node_attributes(road_graph, is_intersection, self.attr_name)
        return road_graph


class RoadNodeIsMajorIntersection(Metric):
    attr_name = cfg.layers.road_nodes.attributes.is_major_intersection
    priority = 4  # before RoadEdgeDistanceToLastMajorIntersection

    def _calculate(self, road_graph: nx.MultiDiGraph, **kwargs) -> nx.MultiDiGraph:
        node_degree = nx.get_node_attributes(
            road_graph, cfg.layers.road_nodes.attributes.node_major_degree
        )
        is_intersection = {node: degree > 2 for node, degree in node_degree.items()}
        nx.set_node_attributes(road_graph, is_intersection, self.attr_name)
        return road_graph


class RoadNodeAverageNeighborDegree(GraphMetric):
    attr_name = cfg.layers.road_nodes.attributes.average_neighbor_degree

    def _calculate(self, road_graph: nx.MultiDiGraph, **kwargs) -> nx.MultiDiGraph:
        avg_neighbor_degree = nx.average_neighbor_degree(road_graph)
        add_node_data(road_graph, avg_neighbor_degree, "average_neighbor_degree")
        return road_graph


class RoadNodeDegree(GraphMetric):
    attr_name = cfg.layers.road_nodes.attributes.node_degree
    priority = 2  # before RoadNodeIsIntersection

    def _calculate(self, road_graph: nx.MultiDiGraph, **kwargs) -> nx.MultiDiGraph:
        for node in road_graph.nodes():
            predecessors = set(road_graph.predecessors(node))
            successors = set(road_graph.successors(node))
            unique_adjacent_nodes = predecessors.union(successors)

            road_graph.nodes[node][self.attr_name] = len(unique_adjacent_nodes)

        return road_graph


class RoadNodeMajorDegree(GraphMetric):
    attr_name = cfg.layers.road_nodes.attributes.node_major_degree
    priority = 2  # before RoadNodeIsMajorIntersection

    def _calculate(self, road_graph: nx.MultiDiGraph, **kwargs) -> nx.MultiDiGraph:
        type_category = nx.get_edge_attributes(
            road_graph, cfg.layers.road_edges.attributes.type_category
        )
        is_major_edge = {
            edge: type_category[edge] in ("major", "highway")
            for edge in road_graph.edges(keys=True)
            if edge in type_category
        }
        for node in road_graph.nodes():
            major_degree = 0
            predecessors = set(road_graph.predecessors(node))
            successors = set(road_graph.successors(node))
            unique_adjacent_nodes = predecessors.union(successors)
            # find which of the unique adjacent_nodes have a street with type_category major
            # either going from or to it
            for adjacent_node in unique_adjacent_nodes:
                edges = ((adjacent_node, node, 0), (node, adjacent_node, 0))
                edges_that_exist = [
                    edge for edge in edges if road_graph.has_edge(*edge)
                ]
                is_major = any(
                    is_major_edge.get(edge, False) for edge in edges_that_exist
                )
                if is_major:
                    major_degree += 1

            road_graph.nodes[node][self.attr_name] = major_degree

        return road_graph


class RoadNodeClustering(GraphMetric):
    attr_name = cfg.layers.road_nodes.attributes.clustering

    def _calculate(self, road_graph: nx.MultiDiGraph, **kwargs) -> nx.MultiDiGraph:
        return momepy.clustering(road_graph, name=self.attr_name)
