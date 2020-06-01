import src.graph_metrics as metric
import networkx as nx


def test_count_graph_nodes_distribution_simple_graph():
    """
    Check if function ``count_graph_nodes_degree_distribution``
    works for simple graph.
    """
    g = nx.Graph()
    g.add_nodes_from([0, 1, 2, 3])
    g.add_edge(0, 1)
    g.add_edge(0, 2)

    degree_distribution = metric.count_graph_nodes_degree_distribution(g)

    assert degree_distribution == [1, 2, 1, 0]


def test_count_average_graph_degree_simple_test():
    """
    Check function ``count_average_graph_degree`` result
    for simple graph.
    """

    g = nx.Graph()
    g.add_nodes_from([0, 1, 2, 3])
    g.add_edge(0, 1)
    g.add_edge(0, 2)

    avg_degree = metric.count_average_graph_degree(g)

    assert avg_degree == 1.0


def test_graph_density():
    """
    Check function ``count_graph_density`` for simple graph.
    """





