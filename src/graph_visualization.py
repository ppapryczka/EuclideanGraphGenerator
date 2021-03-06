import os
from typing import Tuple, List

import matplotlib.pyplot as plt
import networkx as nx


def create_output_directory(
        path: str
) -> str:
    """
    Create directory or do nothing if directory already exists

    Args:
        path: Directory path
    """
    try:
        os.stat(path)
    except:
        os.mkdir(path)
    finally:
        return path


def show_graph(
        graph: nx.Graph,
        filename: None or str
):
    """
    Draw the graph using matplotlib and optionally save it to file.

    Args:
        graph: Graph to visualize
        filename: Name of the file to create or None if file should not be generated
    """
    pos = nx.get_node_attributes(graph, "pos")
    nx.draw_networkx(graph, pos, node_size=4, with_labels=False)
    if filename:
        plt.savefig(filename)
    plt.show()


def show_distribution_chart(
        distribution: List[Tuple[int, int]],
        y_range: None or Tuple[float, float],
        filename: None or str
):
    """
    Draw the distribution chart using matplotlib and optionally save it to file.

    Args:
        distribution: Distribution as list of tuples (key, value)
        y_range: Optional range of y values
        filename: Name of the file to create or None if file should not be generated
    """
    fig, ax = plt.subplots(1, 1)
    for x, value in distribution:
        ax.plot(x, value, 'bo', ms=8)
        ax.vlines(x, 0, value, colors='b', lw=5, alpha=0.5)
    if y_range:
        diff = y_range[1] - y_range[0]
        plt.ylim(y_range[0] - diff * 0.1, y_range[1] + diff * 0.1)
    plt.savefig(filename)
    plt.show()
