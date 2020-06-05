import math
import time
from typing import Sequence, List, Union, Tuple

import networkx as nx
import numpy as np

from src.kd_tree import create_standalone_kd_node, query_pairs, kd_tree


def distance(point1: Sequence[float], point2: Sequence[float]) -> float:
    """
    Count euclidean distance between ``point1`` and ``point2``.

    Args:
        point1: First point as a (x, y).
        point2: Second point as a (x, y).

    Returns:
        Distance as a float number.
    """
    return math.hypot(point2[0] - point1[0], point2[1] - point1[1])


def generate_point_positions(
        low_boundary: float,
        high_boundary: float,
        size: int
) -> Tuple[List[Tuple[float, float]], float]:
    """
    Generate points positions using ``np.random.uniform``
    in range [``low_boundary``, ``high_boundary``).

    Args:
        low_boundary: Low boundary for generation.
        high_boundary: High boundary for generation.
        size: Number of points.

    Returns:
        List of positions as tuples (x,y), execution time
    """
    start_time = time.time()
    positions_x = np.random.uniform(low_boundary, high_boundary, size)
    positions_y = np.random.uniform(low_boundary, high_boundary, size)
    end_time = time.time()
    return list(zip(positions_x, positions_y)), end_time - start_time


def connect_close_enough_edges_(g: nx.Graph, radius: float) -> None:
    """
    Simply iterate over nodes and check distance. Connect nodes if distance
    lower or equal ``radius``. Function modify ``g``!

    Args:
        g: Graph, each node must have ``pos`` attribute.
        radius: Max distance.
    """

    # iterate over nodes and connect - 0(n^2) operation
    for n1 in g.nodes:
        for n2 in range(n1 + 1, len(g.nodes)):
            if distance(g.nodes[n1]["pos"], g.nodes[n2]["pos"]) <= radius:
                g.add_edge(n1, n2)


def generate_simple_random_graph(
        number_of_vertices: int,
        radius: float,
        positions: Union[List, None] = None,
        use_kd_tree: bool = False,
) -> Tuple[nx.Graph, float]:
    """
    Generate positions using ``generate_point_positions`` if ``positions``
    is None, add nodes with positions information, connect nodes closer than
    ``radius``.

    Args:
        number_of_vertices: Number of vertices.
        radius: Minimal distance between nodes.
        positions: Optional positions list.
        use_kd_tree: Use kd tree for connecting nodes. Will iterate through all node pairs if False.

    Returns:
        Random geometric graph, execution time
    """
    start_time = time.time()
    if positions is not None:
        if number_of_vertices != len(positions):
            raise Exception("Number of vertices and length of positions differ.")

    # generate positions if not given
    if positions is None:
        positions, execution_time = generate_point_positions(0.0, 1.0, number_of_vertices)

    # create nodes
    g: nx.Graph = nx.Graph()
    for idx, position in enumerate(positions):
        g.add_node(idx, pos=position)

    if use_kd_tree:
        kd_nodes = list()
        for idx, position in enumerate(positions):
            kd_nodes.append(create_standalone_kd_node(idx, position))
        tree = kd_tree(kd_nodes)
        pairs = query_pairs(tree, radius)
        for i, edge in enumerate(pairs):
            g.add_edge(edge[0], edge[1])
    else:
        connect_close_enough_edges_(g, radius)

    end_time = time.time()
    return g, end_time - start_time


def generate_graphs(
        number_of_graphs: int,
        number_of_vertices: int,
        radius: float,
        use_kd_tree: bool = True,
        print_progress: bool = True
) -> Tuple[List[nx.Graph], float]:
    """
    Count euclidean distance between ``point1`` and ``point2``.

    Args:
        number_of_graphs: Number of graphs to generate.
        number_of_vertices: Number of vertices.
        radius: Minimal distance between nodes.
        use_kd_tree: Use kd tree for connecting nodes. Will iterate through all node pairs if False.
        print_progress: Print generation progress to console

    Returns:
        Distance as a float number.
    """
    start_time = time.time()
    graphs = []
    for i in range(0, number_of_graphs):
        positions, execution_time = generate_point_positions(0.0, 1.0, number_of_vertices)
        g, execution_time = generate_simple_random_graph(
            number_of_vertices=number_of_vertices,
            radius=radius,
            positions=positions,
            use_kd_tree=use_kd_tree
        )
        graphs.append(g)
        if print_progress and i % (number_of_graphs / 10) == 0:
            print("{} %".format(math.ceil(i / number_of_graphs * 100)))
    end_time = time.time()
    generation_time = end_time - start_time
    if print_progress:
        print("100 %")
    return graphs, generation_time
