from networkx import connected_components


def connected_component_subgraphs(G, copy=True):
    """Generate connected components as subgraphs.

    Parameters
    ----------
    G : NetworkX graph
       An undirected graph.

    copy: bool (default=True)
      If True make a copy of the graph attributes

    Returns
    -------
    comp : generator
      A generator of graphs, one for each connected component of G.

    See Also
    --------
    connected_components

    Notes
    -----
    For undirected graphs only.
    Graph, node, and edge attributes are copied to the subgraphs by default.
    """
    for c in connected_components(G):
        if copy:
            yield G.subgraph(c).copy()
        else:
            yield G.subgraph(c)
