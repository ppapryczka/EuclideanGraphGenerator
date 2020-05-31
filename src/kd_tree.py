import math
from collections import namedtuple
from typing import Tuple, List

import numpy


class KdTreeNode(namedtuple('Node', 'id rectangle_min rectangle_max location left_child self_node right_child children')):

    def is_leaf(self):
        return len(self.children) == 1

    def __str__(self):
        return "{}: {} LEFT: {}, RIGHT: {}\n   {}\n    {}" \
            .format(self.id,
                    self.location,
                    self.left_child.id if self.left_child else "None",
                    self.right_child.id if self.right_child else "None",
                    self.left_child if self.left_child else "None",
                    self.right_child if self.right_child else "None")


def create_standalone_kd_node(idx: int,
                              position: Tuple[float, float]) -> KdTreeNode:
    """
    Create standalone node of kd tree. May be used to create initial nodes and then build kd tree from them

    Args:
        idx: Id of the node, id's in the tree must be unique (some edges may not be found otherwise)
        position: Position of the node as (x, y).

    Returns:
        Standalone node of the kd tree.
    """
    return KdTreeNode(id=idx,
                      rectangle_min=numpy.asarray(position).astype(float),
                      rectangle_max=numpy.asarray(position).astype(float),
                      location=position,
                      left_child=None,
                      self_node=None,
                      right_child=None,
                      children=[idx])


def min_distance_rectangle(first: KdTreeNode,
                           second: KdTreeNode) -> float:
    """
    Returns minimal distance between points belonging to node 'first' and 'second'.
    Will return 0 when bounding boxes of nodes are touching or overlapping.
    Otherwise will return shortest possible distance between the node's bounding boxes.
    Order of arguments does not matter.

    Args:
        first: First node. Node contains coordinates of bounding box of belonging nodes.
        second: First node. Node contains coordinates of bounding box of belonging nodes.

    Returns:
        Minimal distance between points in nodes.
    """
    return distance((0., 0.), numpy.maximum(0, numpy.maximum(first.rectangle_min - second.rectangle_max, second.rectangle_min - first.rectangle_max)))


def max_distance_rectangle(first: KdTreeNode, second: KdTreeNode):
    """
    Returns maximal possible distance between points belonging to node 'first' and 'second'.
    Creates bounding box, that contains bounding boxes of both nodes.
    Then returns length of the diagonal of that bounding box.

    Args:
       first: First node. Node contains coordinates of bounding box of belonging nodes.
       second: First node. Node contains coordinates of bounding box of belonging nodes.

    Returns:
       Maximal distance between points in nodes.
    """
    min_p = numpy.minimum(first.rectangle_min, second.rectangle_min)
    max_p = numpy.maximum(first.rectangle_max, second.rectangle_max)
    return distance(min_p, max_p)


def distance(point1: Tuple[float, float],
             point2: Tuple[float, float]) -> float:
    """
    Count euclidean distance between ``point1`` and ``point2``.

    Args:
        point1: First point as a (x, y).
        point2: Second point as a (x, y).

    Returns:
        Distance as a float number.
    """
    return math.hypot(point2[0] - point1[0], point2[1] - point1[1])


def query_pairs(tree: KdTreeNode, radius: float):
    """
    Recurrently checks existence of all possible edges of graph.

    Args:
        tree: Root node of kd tree
        radius: Radius. When distance between nodes is smaller, edge exists.

    Returns:
        Set of graph edges (as (node_id_1, node_id_2) pairs).
    """
    results = set()

    def check_recursively(node1: KdTreeNode, node2: KdTreeNode):
        if node1 is None or node2 is None:
            # Some nodes, especially leaves, may not have left and/or right child
            return
        if min_distance_rectangle(node1, node2) >= radius:
            # If minimal distance is bigger than radius, there is no point checking further
            return
        elif max_distance_rectangle(node1, node2) < radius:
            # If maximum distance is within limit, edges exist between all pairs of node's children (including nodes themselves)
            for n1 in node1.children:
                for n2 in node2.children:
                    if n1 < n2:
                        results.add((n1, n2))
            return
        elif node1.is_leaf():
            # Node 1 is leaf
            # Not entire Node 2 is within radius limit
            # Node 2 must be split
            check_recursively(node1, node2.self_node)
            check_recursively(node1, node2.left_child)
            check_recursively(node1, node2.right_child)
        elif node2.is_leaf():
            # Node 2 is leaf
            # Not entire Node 1 is within radius limit
            # Node 1 must be split
            check_recursively(node2, node1.self_node)
            check_recursively(node2, node1.left_child)
            check_recursively(node2, node1.right_child)
        else:
            # Both nodes are not leaves
            # Split node 2 (could also be implemented the other way round)
            check_recursively(node1, node2.self_node)
            check_recursively(node1, node2.left_child)
            check_recursively(node1, node2.right_child)

    check_recursively(tree, tree)
    return results


def kd_tree(nodes: List[KdTreeNode],
            depth: int = 0):
    """
    Recurrently checks existence of all possible edges of graph.

    Args:
        nodes: List of nodes to be placed in subtree
        depth: Depth level of the tree, used to determine which axis to sort by.

    Returns:
        Kd tree containing given nodes
    """
    if len(nodes) == 0:
        return None

    # In kd tree nodes are alternately sorted by x and y axis
    axis = depth % 2
    nodes.sort(key=lambda x: x.location[axis])
    # Choose median node
    median = len(nodes) // 2

    # Calculate bounding box of median node and all other nodes
    node_locations = list(map(lambda x: x.location, nodes))
    minimums = list(map(min, zip(*node_locations)))
    nmins = numpy.asarray((float(minimums[0]), float(minimums[1])))
    maximums = list(map(max, zip(*node_locations)))
    nmaxes = numpy.asarray((float(maximums[0]), float(maximums[1])))

    return KdTreeNode(
        id=nodes[median].id,
        rectangle_min=numpy.asarray(nmins),
        rectangle_max=numpy.asarray(nmaxes),
        location=nodes[median].location,
        left_child=kd_tree(nodes[:median], depth + 1),
        self_node=nodes[median],
        right_child=kd_tree(nodes[median + 1:], depth + 1),
        children=list(map(lambda x: x.id, nodes))
    )
