import argparse
from typing import List
import sys
from src.generate_random_graph import generate_simple_random_graph
import networkx as nx
import matplotlib.pyplot as plt


def check_radius(parser: argparse.ArgumentParser, radius: float):
    try:
        x = float(radius)
    except ValueError:
        parser.error(
            "Not appropriate value for radius! Expected float between (0.0, 1.0)."
        )

    if 0.0 < x < 1.0:
        return x
    else:
        parser.error(
            "Not appropriate value for radius! Expected float between (0.0, 1.0)."
        )


def generate_random_graph_command(command_args: List[str]):
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

    args = parser.parse_args(command_args)

    g = generate_simple_random_graph(args.num, args.radius)
    pos = nx.get_node_attributes(g, "pos")
    nx.draw_networkx(g, pos, node_size=10, with_labels=False)
    plt.show()


if __name__ == "__main__":
    generate_random_graph_command(sys.argv[1:])
