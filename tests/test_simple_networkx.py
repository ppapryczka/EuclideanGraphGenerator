import networkx as nx


def test_graph_networkx_nodes_num_check():
    g: nx.Graph = nx.Graph()
    g.add_node(1)
    assert len(g.nodes) == 1
