import math
from typing import List

import networkx as nx
from networkx.algorithms.assortativity.tests.test_correlation import np
from scipy.stats import binom

from networkx_extensions import connected_component_subgraphs


class GraphsComponentsStatistics:
    def __init__(
            self,
            mean_number_of_components,
            mean_number_of_nodes_in_component,
            mean_number_of_edges_in_component,
            mean_density_of_component,
            mean_number_of_trees_per_graph,
            mean_size_of_tree
    ):
        self.mean_number_of_components = mean_number_of_components
        self.mean_number_of_nodes_in_component = mean_number_of_nodes_in_component
        self.mean_number_of_edges_in_component = mean_number_of_edges_in_component
        self.mean_density_of_component = mean_density_of_component
        self.mean_number_of_trees_per_graph = mean_number_of_trees_per_graph
        self.mean_size_of_tree = mean_size_of_tree


class GraphsCycleRelatedStatistics:
    def __init__(
            self,
            mean_number_of_base_cycles,
            mean_length_of_base_cycle
    ):
        self.mean_number_of_base_cycles = mean_number_of_base_cycles
        self.mean_length_of_base_cycle = mean_length_of_base_cycle


def calculate_expected_mean_degree(
        number_of_vertices: int,
        radius: float
) -> float:
    """
    Calculate expected mean degree.

    Args:
        number_of_vertices: Number of vertices.
        radius: Minimal distance between nodes.

    Returns:
        Expected mean degree.
    """
    return np.pi * radius * radius * (number_of_vertices - 1)


def calculate_real_mean_degree(
        graphs: List[nx.Graph],
) -> float:
    """
    Calculate real mean degree.

    Args:
        graphs: List of graphs

    Returns:
        Real mean degree.
    """
    number_of_vertices = graphs[0].number_of_nodes()
    for g in graphs:
        if g.number_of_nodes() != number_of_vertices:
            raise Exception("All graphs must have the same number of nodes")
    averages = list()
    for i, graph in enumerate(graphs):
        average_graph_degree = sum(map(lambda x: x[1], graph.degree)) / number_of_vertices
        averages.append(average_graph_degree)
    return np.average(averages)


def calculate_expected_number_of_edges(
        number_of_vertices: int,
        radius: float
) -> float:
    """
    Calculate expected number of edges.

    Args:
        number_of_vertices: Number of vertices.
        radius: Minimal distance between nodes.

    Returns:
        Expected number of edges.
    """
    return np.pi * radius * radius * number_of_vertices * (number_of_vertices - 1) / 2


def calculate_real_number_of_edges(
        graphs: List[nx.Graph]
) -> float:
    """
    Calculate real number of edges.

    Args:
        graphs: List of graphs

    Returns:
        Real number of edges.
    """
    averages = list()
    for index, graph in enumerate(graphs):
        averages.append(len(graph.edges))
    return np.average(averages)


def calculate_expected_density(
        number_of_vertices: int,
        radius: float
) -> float:
    """
    Calculate expected density.

    Args:
        number_of_vertices: Number of vertices.
        radius: Minimal distance between nodes.

    Returns:
        Expected density.
    """
    return np.pi * radius * radius * number_of_vertices * (number_of_vertices - 1) / 2 / ((number_of_vertices * (number_of_vertices - 1)) / 2)


def calculate_real_density(
        graphs: List[nx.Graph],
) -> float:
    """
    Calculate real density.

    Args:
        graphs: List of graphs

    Returns:
        Real density.
    """
    number_of_vertices = graphs[0].number_of_nodes()
    for g in graphs:
        if g.number_of_nodes() != number_of_vertices:
            raise Exception("All graphs must have the same number of nodes")
    averages = list()
    for index, graph in enumerate(graphs):
        averages.append(len(graph.edges))
    return np.average(averages) / ((number_of_vertices * (number_of_vertices - 1)) / 2)


def calculate_approximation_error(
        real_value: float,
        expected_value: float
) -> float:
    """
    Calculate approximation error.

    Args:
        real_value: Real value.
        expected_value: Expected value.

    Returns:
        Approximation error.
    """
    return float("-inf") if real_value == 0 else (real_value - expected_value) / real_value * 100


def calculate_expected_degree_distribution(
        number_of_vertices: int,
        radius: float,
) -> List:
    """
    Calculate expected degree distribution.

    Args:
        number_of_vertices: Number of vertices.
        radius: Minimal distance between nodes.

    Returns:
        Expected degree distribution.
    """
    predicted_probability = min(np.pi * radius * radius, 1)
    max_to_check = min(math.ceil(2 * (number_of_vertices * predicted_probability)) + 10, number_of_vertices)
    range_to_check = range(0, max_to_check)

    theoretical_values = []
    for x in range_to_check:
        value = number_of_vertices * binom.pmf(x, number_of_vertices - 1, predicted_probability)
        theoretical_values.append((x, value))

    return theoretical_values


def calculate_real_degree_distribution(
        number_of_vertices: int,
        radius: float,
        graphs: List[nx.Graph],
) -> List:
    """
    Calculate real degree distribution.

    Args:
        number_of_vertices: Number of vertices.
        radius: Minimal distance between nodes.
        graphs: List of graphs

    Returns:
        Real degree distribution.
    """
    number_of_graphs = len(graphs)
    predicted_probability = min(np.pi * radius * radius, 1)
    max_to_check = min(math.ceil(2 * (number_of_vertices * predicted_probability)) + 10, number_of_vertices)
    range_to_check = range(0, max_to_check)

    degrees = [0] * number_of_vertices
    for index, graph in enumerate(graphs):
        for degree in map(lambda x: x[1], graph.degree):
            degrees[degree] += 1

    real_mean_degree_distribution = list(map(lambda degree: degree / number_of_graphs, degrees))

    real_values = []
    for x in range_to_check:
        value = real_mean_degree_distribution[x]
        real_values.append((x, value))

    return real_values


def calculate_graph_components_statistics(
        graphs: List[nx.Graph],
) -> GraphsComponentsStatistics:
    """
    Calculate component-related statistics.

    Args:
        graphs: List of graphs

    Returns:
        Components-related graph stats
    """
    number_of_graphs = len(graphs)
    connected_components_per_graph = list(map(lambda graph: list(connected_component_subgraphs(graph)), graphs))
    all_connected_components = [component for sublist in connected_components_per_graph for component in sublist]
    numbers_of_components_per_graph = list(map(lambda components_per_graph: len(components_per_graph), connected_components_per_graph))

    all_trees_amongst_components = [component for component in all_connected_components if nx.is_tree(component)]
    number_of_all_trees = len(all_trees_amongst_components)

    densities_of_components = []
    for component in all_connected_components:
        if component.number_of_nodes() == 1:
            densities_of_components.append(1)
        else:
            density = 2 * component.number_of_edges() / (component.number_of_nodes() * (component.number_of_nodes() - 1))
            densities_of_components.append(density)

    mean_number_of_components = np.average(numbers_of_components_per_graph)
    mean_number_of_nodes_in_component = np.average(list(map(lambda component: component.number_of_nodes(), all_connected_components)))
    mean_number_of_edges_in_component = np.average(list(map(lambda component: component.number_of_edges(), all_connected_components)))
    mean_density_of_component = np.average(densities_of_components)
    mean_number_of_trees_per_graph = number_of_all_trees / number_of_graphs
    if number_of_all_trees == 0:
        mean_size_of_tree = 0
    else:
        mean_size_of_tree = sum(tree.number_of_nodes() for tree in all_trees_amongst_components) / number_of_all_trees

    return GraphsComponentsStatistics(
        mean_number_of_components,
        mean_number_of_nodes_in_component,
        mean_number_of_edges_in_component,
        mean_density_of_component,
        mean_number_of_trees_per_graph,
        mean_size_of_tree
    )


def calculate_graph_cycle_related_statistics(
        graphs: List[nx.Graph],
) -> GraphsCycleRelatedStatistics:
    """
    Calculate cycle-related statistics.

    Args:
        graphs: List of graphs

    Returns:
        Cycle-related graph stats
    """
    number_of_graphs = len(graphs)
    base_cycles_per_graph = list(map(lambda graph: nx.cycle_basis(graph), graphs))
    lengths_of_base_cycles = [
        len(cycle) for sublist in base_cycles_per_graph for cycle in sublist
    ]
    mean_number_of_base_cycles = len(lengths_of_base_cycles) / number_of_graphs
    mean_length_of_base_cycle = 0 if len(lengths_of_base_cycles) == 0 else np.average(lengths_of_base_cycles)
    return GraphsCycleRelatedStatistics(mean_number_of_base_cycles, mean_length_of_base_cycle)


def calculate_mean_clustering(
        graphs: List[nx.Graph],
) -> GraphsCycleRelatedStatistics:
    """
    Calculate mean clustering.

    Args:
        graphs: List of graphs

    Returns:
        Mean clustering
    """
    clustering = list()
    for graph in graphs:
        clustering.append(nx.average_clustering(graph))
    print("Średni współczynnik klasteryzacji: {}".format(np.average(clustering)))
    return np.average(clustering)
