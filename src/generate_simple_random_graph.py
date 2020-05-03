import math
import time
from typing import Sequence, List, Union

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from kd_tree import create_standalone_kd_node, query_pairs, kd_tree


def distance(point1: Sequence[float],
             point2: Sequence[float]) -> float:
    """
    Count euclidean distance between ``point1`` and ``point2``.

    Args:
        point1: First point as a (x, y).
        point2: Second point as a (x, y).

    Returns:
        Distance as a float number.
    """
    return math.hypot(point2[0] - point1[0], point2[1] - point1[1])


def generate_point_positions(low_boundary: float,
                             high_boundary: float,
                             size: int) -> List:
    """
    Generate points positions using ``np.random.uniform``
    in range [``low_boundary``, ``high_boundary``).

    Args:
        low_boundary: Low boundary for generation.
        high_boundary: High boundary for generation.
        size: Number of points.

    Returns:
        List of positions as tuples (x,y).
    """
    positions_x = np.random.uniform(low_boundary, high_boundary, size)
    positions_y = np.random.uniform(low_boundary, high_boundary, size)
    return list(zip(positions_x, positions_y))


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


def generate_simple_random_graph(vertices_number: int,
                                 radius: float,
                                 positions: Union[List, None] = None,
                                 use_kd_tree: bool = False) -> nx.Graph:
    """
    Generate positions using ``generate_point_positions`` if ``positions``
    is None, add nodes with positions information, connect nodes closer than
    ``radius``.

    Args:
        vertices_number: Number of vertices.
        radius: Minimal distance between nodes.
        positions: Optional positions list.
        use_kd_tree: Use kd tree for connecting nodes. Will iterate through all node pairs if False.

    Returns:
        Random geometric graph.
    """

    if positions is not None:
        if vertices_number != len(positions):
            Exception("Number of vertices and length of positions differ.")

    # generate positions if not given
    if positions is None:
        positions = generate_point_positions(0.0, 1.0, vertices_number)

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

    return g

    """
    # plot points
    positions_x = list((p[0] for p in positions))
    positions_y = list((p[1] for p in positions))

    print(positions_x, positions_y)

    plt.scatter(positions_x, positions_y)
    plt.show()
    # plot points
    """


if __name__ == "__main__":
    n = 10000
    r = 0.1

    positions = generate_point_positions(0., 1., n)

    start = time.time()
    g_kd = generate_simple_random_graph(n, r, positions=positions, use_kd_tree=True)
    end = time.time()
    time_kd = end - start

    start = time.time()
    g_n2 = generate_simple_random_graph(n, r, positions=positions, use_kd_tree=False)
    end = time.time()
    time_n2 = end - start

    print("Number of nodes: {}".format(n))
    print("With radius:     {}".format(r))
    print("With kd tree:    {} seconds".format(time_kd))
    print("With n^2 search: {} seconds".format(time_n2))
    print("With kd edges:   {}".format(len(g_kd.edges)))
    print("With n^2 edges:  {}".format(len(g_n2.edges)))

    pos = nx.get_node_attributes(g_kd, "pos")
    nx.draw_networkx(g_kd, pos, node_size=4, with_labels=False)
    plt.show()

    pos = nx.get_node_attributes(g_n2, "pos")
    nx.draw_networkx(g_n2, pos, node_size=4, with_labels=False)
    plt.show()
