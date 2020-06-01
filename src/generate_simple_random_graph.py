import math
import time
from datetime import datetime
from typing import Sequence, List, Union

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from scipy.stats import binom

from kd_tree import create_standalone_kd_node, query_pairs, kd_tree
from networkx_extensions import connected_component_subgraphs


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

    # -------------------------------------------------------------------------------------------------
    # Test grapg generation

    # Number of vertices
    ns = [10]
    # Radiuses
    rs = [0.1]

    for r in rs:
        for n in ns:
            print("-------------------------------------------------------------------")
            print("Liczba wierzchołków: {}".format(n))
            print("Zasięg:              {}".format(r))

            # Generate point positions
            positions = generate_point_positions(0., 1., n)

            # Generate Euclidean graph using kd tree and measure time
            start = time.time()
            g_kd = generate_simple_random_graph(n, r, positions=positions, use_kd_tree=True)
            end = time.time()
            time_kd = end - start
            print("Z drzewem kd:        {} s".format(time_kd))
            print("Liczba krawędzi:     {}".format(len(g_kd.edges)))

            # Generate Euclidean graph not using kd tree and measure time
            start = time.time()
            g_n2 = generate_simple_random_graph(n, r, positions=positions, use_kd_tree=False)
            end = time.time()
            time_n2 = end - start
            print("Z przeszukaniem n^2: {} s".format(time_n2))
            print("Liczba krawędzi:     {}".format(len(g_n2.edges)))

            # Draw graph to file
            pos = nx.get_node_attributes(g_n2, "pos")
            figure = nx.draw_networkx(g_n2, pos, node_size=4, with_labels=False)
            # plt.show()
            plt.savefig("{}_{}_{}.png".format(datetime.now(), n, r))
            plt.show()

    # -------------------------------------------------------------------------------------------------
    # Checking statistical properties of graphs
    number_of_graphs = 100
    n = 100
    r = 0.1
    graphs = list()

    print("-------------------------------------------------------------------")
    print("Sprawdzanie właściwości statystycznych")
    print("Liczba grafów:        {}".format(number_of_graphs))
    print("Liczba wierzchołków:  {}".format(n))
    print("Zasięg:               {}".format(r))

    print("-------------------------------------------------------------------")
    print("Generacja grafów ...")
    start = time.time()
    for i in range(0, number_of_graphs):
        positions = generate_point_positions(0., 1., n)
        g_kd = generate_simple_random_graph(n, r, positions=positions, use_kd_tree=True)
        graphs.append(g_kd)
        if i % (number_of_graphs / 100) == 0:
            print("{} %".format(math.ceil(i / number_of_graphs * 100)))
    end = time.time()
    time = end - start
    print("100 %")
    print("Generacja grafów ukończona w {} sekundy".format(time))

    print("-------------------------------------------------------------------")
    print("Sprawdzanie średniego stopnia wierzchołka ...")
    averages = list()
    for index, graph in enumerate(graphs):
        average_graph_degree = sum(map(lambda x: x[1], graph.degree)) / n
        averages.append(average_graph_degree)
    real = np.average(averages)
    predicted = np.pi * r * r * (n - 1)
    error = float("-inf") if real == 0 else (real - predicted) / real * 100
    print("Rzeczywisty średni stopień    {}".format(real))
    print("Przewidywany średni stopień   {}".format(predicted))
    print("Błąd                          {} %".format(error))

    print("-------------------------------------------------------------------")
    print("Sprawdzanie ilości krawędzi ...")
    averages = list()
    for index, graph in enumerate(graphs):
        averages.append(len(graph.edges))
    real = np.average(averages)
    predicted = np.pi * r * r * n * (n - 1) / 2
    error = float("-inf") if real == 0 else (real - predicted) / real * 100
    print("Rzeczywista ilość krawędzi    {}".format(real))
    print("Przewidywana ilość krawędzi   {}".format(predicted))
    print("Błąd                          {} %".format(error))

    print("-------------------------------------------------------------------")
    print("Sprawdzanie gęstości grafu ...")
    averages = list()
    for index, graph in enumerate(graphs):
        averages.append(len(graph.edges))
    real = np.average(averages) / ((n * (n - 1)) / 2)
    predicted = np.pi * r * r * n * (n - 1) / 2 / ((n * (n - 1)) / 2)
    error = float("-inf") if real == 0 else (real - predicted) / real * 100
    print("Rzeczywista gęstość           {}".format(real))
    print("Przewidywana gęstość          {}".format(predicted))
    print("Błąd                          {} %".format(error))

    print("-------------------------------------------------------------------")
    print("Sprawdzanie rozkładu stopni wierzchołków ...")
    # Create array with value "0" for all possible degree values
    predicted_propability = min(np.pi * r * r, 1)
    degrees = [0] * n
    # Count degree value occurrences in all graphs
    for index, graph in enumerate(graphs):
        for degree in map(lambda x: x[1], graph.degree):
            degrees[degree] += 1
    # Normalize (divide by the number of graphs and the number of vertices in graph
    mean_degrees = list(map(lambda degree: degree / number_of_graphs, degrees))
    print("Rozkład stopni wierzchołków ...")
    print(mean_degrees)

    print("-------------------------------------------------------------------")
    print("Generowanie wykresu rozkładu rzeczywistego ...")
    fig, ax = plt.subplots(1, 1)
    range_to_check = min(math.ceil(2 * (n * predicted_propability)) + 10, n)
    for x in range(0, range_to_check):
        # Plot point on chart
        ax.plot(x, mean_degrees[x], 'bo', ms=8, label='binom pmf')
        # Plot blue line from x axis to point
        ax.vlines(x, 0, mean_degrees[x], colors='b', lw=5, alpha=0.5)
    plt.savefig("chart_real_{}_{}_{}_{}.png".format(datetime.now(), n, r, number_of_graphs))
    plt.show()

    print("-------------------------------------------------------------------")
    print("Generowanie wykresu rozkładu teoretycznego ...")
    fig, ax = plt.subplots(1, 1)
    for x in range(0, range_to_check):
        ax.plot(x, n * binom.pmf(x, n - 1, predicted_propability), 'bo', ms=8, label='binom pmf')
        ax.vlines(x, 0, n * binom.pmf(x, n - 1, predicted_propability), colors='b', lw=5, alpha=0.5)
    plt.savefig("chart_theoretical_{}_{}_{}_{}.png".format(datetime.now(), n, r, number_of_graphs))
    plt.show()

    print("-------------------------------------------------------------------")
    print("Sprawdzanie liczby składowych spójnych ...")

    connected_components_per_graph = list(map(lambda graph: list(connected_component_subgraphs(graph)), graphs))
    connected_components = [component for sublist in connected_components_per_graph for component in sublist]

    number_of_trees = sum(1 if nx.is_tree(component) else 0 for component in connected_components)
    mean_number_of_trees = number_of_trees / n
    mean_size_of_tree = 0 if number_of_trees == 0 else sum(component.number_of_nodes() if nx.is_tree(component) else 0 for component in connected_components) / number_of_trees

    mean_number_of_nodes_in_component = np.average(list(map(lambda component: component.number_of_nodes(), connected_components)))
    mean_number_of_edges_in_component = np.average(list(map(lambda component: component.number_of_edges(), connected_components)))

    densities_of_components = list(map(lambda component: 1 if component.number_of_nodes() == 1 else 2 * component.number_of_edges() / (component.number_of_nodes() * (component.number_of_nodes() - 1)), connected_components))
    mean_density_of_component = np.average(densities_of_components)

    numbers_of_components = list(map(lambda graph: len(list(connected_component_subgraphs(graph))), graphs))
    mean_number_of_components = np.average(numbers_of_components)

    print("Średnia liczba składowych spójnych:     {}".format(mean_number_of_components))
    print("Średnia liczba wierzchołków składowej:  {}".format(mean_number_of_nodes_in_component))
    print("Średnia liczba krawędzi składowej:      {}".format(mean_number_of_edges_in_component))
    print("Średnia gęstość składowej:              {}".format(mean_density_of_component))
    print("Średnia liczba drzew w grafie:          {}".format(mean_number_of_trees))
    print("Średnia liczba wierzchołków w drzewie:  {}".format(mean_size_of_tree))

    print("-------------------------------------------------------------------")
    print("Sprawdzanie liczby cykli ...")
    base_cycles_for_graphs = list(map(lambda graph: nx.cycle_basis(graph), graphs))
    lengths_of_base_cycles = [len(cycle) for sublist in base_cycles_for_graphs for cycle in sublist]
    mean_number_of_base_cycles = len(lengths_of_base_cycles) / number_of_graphs
    mean_length_of_base_cycle = 0 if len(lengths_of_base_cycles) == 0 else np.average(lengths_of_base_cycles)
    print("Liczba cykli bazowych: {}".format(mean_number_of_base_cycles))
    print("Średnia długość cyklu: {}".format(mean_length_of_base_cycle))
