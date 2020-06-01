"""
Module with function to run euclidean graph generation
from command line.
"""

import argparse
from typing import List
import sys
from src.generate_random_graph import generate_simple_random_graph
import networkx as nx
import matplotlib.pyplot as plt


def check_radius(parser: argparse.ArgumentParser, radius: str) -> float:
    """
    Check if given ``radius`` is float number in range
    (0.0, 1.0). If not float or bad value call parser error.

    Args:
        parser: Command parser.
        radius: String to check if appropriate radius.

    Returns:
        Radius as float.
    """
    # try if it is float number
    try:
        x = float(radius)
    except ValueError:
        parser.error(
            "Not appropriate value for radius! Expected float between (0.0, 1.0)."
        )

    # check if radius have appropriate value
    if 0.0 < x < 1.0:
        return x
    else:
        parser.error(
            "Not appropriate value for radius! Expected float between (0.0, 1.0)."
        )


def generate_random_graph_command(command_args: List[str]) -> None:
    """
    Parse ``command_args`` and run simple generation with
    given number of vertices and radius. If given output
    file name save generate graph as png file

    Args:
        command_args: Argument to command.
    """

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--num", "-N", help="number of vertices", type=int, required=True
    )

    parser.add_argument(
        "--radius",
        "-R",
        help="radius - max distance between nodes",
        type=lambda x: check_radius(parser, x),
        required=True,
    )

    parser.add_argument(
        "--output",
        "-O",
        help="name of output file with graph",
        required=False,
        default=None,
    )

    # parse arguments
    args = parser.parse_args(command_args)

    # generate simple graph with given number of vertices and radius
    g = generate_simple_random_graph(args.num, args.radius)

    # draw graph
    pos = nx.get_node_attributes(g, "pos")
    nx.draw_networkx(g, pos, node_size=10, with_labels=False)

    # check if output argument given and save or show
    if args.output is None:
        plt.show()
    else:
        plt.savefig(args.output, format="png", dpi=300)


if __name__ == "__main__":
    generate_random_graph_command(sys.argv[1:])
