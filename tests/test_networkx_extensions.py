from src.networkx_extensions import connected_component_subgraphs
import networkx as nx


def test_generate_connected_component_subgraphs():
    """
    Check subgraphs generate by connected_component_subgrahs.
    """
    G = nx.path_graph(4)

    G.add_edge(5, 6)
    graphs = list(connected_component_subgraphs(G))

    assert sorted(graphs[0].nodes) == [0, 1, 2, 3]
    assert sorted(graphs[1].nodes) == [5, 6]


def test_generate_connected_component_subgraphs_copy():
    """
    Check subgraphs generate by connected_component_subgrahs.
    Copy set to True.
    """

    G = nx.path_graph(4)

    G.add_edge(5, 6)
    graphs = list(connected_component_subgraphs(G, copy=False))

    assert sorted(graphs[0].nodes) == [0, 1, 2, 3]
    assert sorted(graphs[1].nodes) == [5, 6]
