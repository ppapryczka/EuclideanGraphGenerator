from src.generate_random_graph import (
    distance,
    generate_point_positions,
    generate_simple_random_graph,
)
import math
import pytest
import networkx as nx


def test_distance_1():
    """
    Simple test for distance function.
    """
    r: float = distance([1, 1], [1, 2])
    assert r == 1


def test_distance_2():
    """
    Simple test for distance function.
    """
    r: float = distance([1, 1], [3, 3])
    assert r == math.sqrt(8)


def test_generate_point_positions_1():
    """
    Test if points are in correct range
    and number of points is appropriate.
    """
    points = generate_point_positions(0.0, 1.0, 10)

    for p in points:
        assert 0.0 <= p[0] <= 1.0
        assert 0.0 <= p[0] <= 1.0

    assert len(points) == 10


def test_generate_point_positions_2():
    """
    Test if points are in correct range
    and number of points is appropriate.
    """
    points = generate_point_positions(0.5, 1.5, 7)

    for p in points:
        assert 0.5 <= p[0] <= 1.5
        assert 0.5 <= p[0] <= 1.5

    assert len(points) == 7


def test_generate_simple_random_graph_differ_vertices_num():
    """
    Test if exception raise if vertices number is
    different than length of points list.
    """
    points = generate_point_positions(0.0, 1.0, 10)

    with pytest.raises(Exception):
        generate_simple_random_graph(5, 0.5, points)


def test_generate_simple_random_graph_zero_connections():
    """
    Check if non of nodes are connected.
    """

    points = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]

    graph = generate_simple_random_graph(4, 0.0, points)

    for n in graph.nodes:
        assert graph.degree[n] == 0
    assert len(graph.nodes) == 4


def test_generate_simple_random_graph_all_connections():
    """
    Check if all nodes are connected.
    """

    points = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]

    graph = generate_simple_random_graph(4, 3, points)

    for n in graph.nodes:
        assert graph.degree[n] == 3
    assert len(graph.nodes) == 4


def test_generate_same_graph_with_kd_tree_1():
    """
    Check if kd tree generate same graph.
    """

    points = generate_point_positions(0.0, 1.0, 10)

    g1: nx.Graph = generate_simple_random_graph(10, 0.3, points)
    g2: nx.Graph = generate_simple_random_graph(10, 0.3, points, True)

    for n in g1.nodes:
        assert sorted(list(g1.edges(n))) == sorted(list(g2.edges(n)))


def test_generate_same_graph_with_kd_tree_2():
    """
    Check if kd tree generate same graph.
    """
    points = generate_point_positions(0.0, 1.0, 15)

    g1: nx.Graph = generate_simple_random_graph(15, 0.3, points)
    g2: nx.Graph = generate_simple_random_graph(15, 0.3, points, True)

    for n in g1.nodes:
        assert sorted(list(g1.edges(n))) == sorted(list(g2.edges(n)))


def test_generate_graph_with_no_given_vertices_list_1():
    """
    Check number of nodes in graph if no vertices list is
    given.
    """
    g: nx.Graph = generate_simple_random_graph(10, 0.1)

    assert len(list(g.nodes)) == 10


def test_generate_graph_with_no_given_vertices_list_2():
    """
    Check number of nodes in graph if no vertices list is
    given. Check if all nodes are connected
    """
    g: nx.Graph = generate_simple_random_graph(10, 2)

    for n in g.nodes:
        assert len(list(g.edges(n))) == 9
