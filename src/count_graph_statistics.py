import src.graph_metrics as metric
from src.generate_random_graph import (
    generate_point_positions,
    generate_simple_random_graph,
)
from typing import Tuple, List
import networkx as nx
import time
import numpy as np
import matplotlib.pyplot as plt


def generate_n_graphs(
    n: int, vertices_number: int, radius: float
) -> Tuple[List[nx.Graph], float]:
    """
    Generate ``n`` random euclidean graphs with ``vertices_number``
    nodes and radius equal ``radius``.

    Args:
        n: Number of graphs.
        vertices_number: Number of vertices for graphs.
        radius: Maximal distance between nodes.

    Returns:
        Tuple - generated graphs and execution time.
    """

    start_time = time.time()
    graphs = []

    for i in range(n):
        points = generate_point_positions(0.0, 1.0, vertices_number)
        g = generate_simple_random_graph(
            vertices_number, radius, points, use_kd_tree=True
        )
        graphs.append(g)
    time_diff = time.time() - start_time

    return graphs, time_diff


def plot_graph_degree():
    g = generate_simple_random_graph(3000, 0.1, use_kd_tree=True)
    degree_distribution = metric.count_graph_nodes_degree_distribution(g)

    fig, ax = plt.subplots(1, 1)
    for x in range(0, range_to_check):
        # plot point on chart
        ax.plot(x, degree_distribution[x], "bo", ms=8, label="binom pmf")
        # plot blue line from x axis to point
        ax.vlines(x, 0, degree_distribution[x], colors="b", lw=5, alpha=0.5)
    plt.show()


if __name__ == "__main__":
    plot_graph_degree()
