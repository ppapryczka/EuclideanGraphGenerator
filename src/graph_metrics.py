from src.generate_random_graph import generate_point_positions, generate_simple_random_graph
from typing import List, Tuple
import networkx as nx
import time
import numpy as np
from scipy.stats import binom
import math


def count_average_graph_degree(g: nx.Graph) -> float:
    """
    Count average graph degree for graph ``g``.

    Args:
        g: Graph.

    Returns:
        Graph ``g`` average graph degree.
    """

    average_graph_degree = sum(map(lambda x: x[1], g.degree)) / len(g.nodes)
    return average_graph_degree


def count_expected_graph_degree(vertices_num: int, radius: float) -> float:
    """
    Count expected euclidean graph degree.

    Args:
        vertices_num: Number of vertices.
        radius: Maximal distance between nodes.

    Returns:
        Expected degree.
    """
    return np.pi * radius * radius * (vertices_num - 1)


def count_expected_graph_edges_number(vertices_num: int, radius: float) -> float:
    """
    Count expected euclidean edges number.

    Args:
        vertices_num: Number of vertices.
        radius: Maximal distance between nodes.

    Returns:
        Expected number of edges.
    """
    return np.pi * radius * radius * vertices_num * (vertices_num - 1) / 2


def count_graph_density(g: nx.Graph) -> float:
    """
    Count ``g`` graph density.

    Args:
        g: Graph.

    Returns:
        Density of graph ``g``.
    """
    # get number of nodes
    n = len(g.nodes)
    # return density
    return len(g.edges)/((n * (n - 1)) / 2)


def count_expected_graph_density(vertices_num: int, radius: float) -> float:
    """
    Count expected euclidean graph density.

    Args:
        vertices_num: Number of vertices.
        radius: Maximal distance between nodes.

    Returns:
        Expected graph density.
    """
    return np.pi * radius * radius * vertices_num * (vertices_num - 1) / 2 / ((vertices_num * (vertices_num - 1)) / 2)


def count_graph_nodes_degree_distribution(g: nx.Graph) -> List[int]:
    """
    Count graph ``g`` nodes degree distribution.

    Args:
        g: Graph.

    Returns:
        Degree distribution of ``g``.
    """
    # Create array with value "0" for all possible degree values (max value is n-1)
    degrees = [0] * len(g.nodes)

    for degree in map(lambda x: x[1], g.degree):
        degrees[degree] += 1

    return degrees


def count_range_to_compare_degrees(vertices_num: int, radius: float) -> int:
    """
    Count range limit to compare degrees.

    Args:
        vertices_num: Number of vertices.
        radius: Maximal distance between nodes.

    Returns:
         High boundary of degree comparision.
    """

    predicted_probability = min(np.pi * radius * radius, 1)
    max_to_check = min(math.ceil(2 * (vertices_num * predicted_probability)) + 10, vertices_num)
    return max_to_check


def count_expected_graph_degree_distribution(range_of_degree: int, vertices_number: int, radius: float) -> List:
    """
    Count expected graph degrees distribution.

    Args:
        range_of_degree: High boundary of distribution.
        vertices_number: Number of vertices.
        radius: Maximal distance between nodes.

    Returns:
        List of expected degree boundaries.
    """

    predicted_probability = min(np.pi * radius * radius, 1)
    degree_distribution = []
    for x in range(0, range_of_degree):
        value = vertices_number * binom.pmf(x, vertices_number - 1, predicted_probability)
        degree_distribution.append((x, value))
    return degree_distribution



def count_component_stats():
    pass