from collections.abc import Iterable, Iterator
from functools import reduce

import networkx as nx
from bitarray import frozenbitarray as fbarray, bitarray
from bitarray.util import zeros as bazeros
from typing import TypeVar
from itertools import combinations

#from .abstract_ps import AbstractPS
from paspailleur.pattern_structures.abstract_ps import AbstractPS

NodeType = TypeVar('NodeType')


class GraphPS(AbstractPS):
    """A pattern structure to work with graphs

    Common description for a set of graphs is their maximal common subgraph
    (i.e. maximal common subset of edges).
    The graphs are intrinsically represented by their adjacency matrices stored as a list of bitarrays.
    """
    PatternType = tuple[fbarray, ...]
    bottom: PatternType  # Graph, containing all possible edges
    nodes: list[NodeType]  # List of all nodes

    def __init__(self, nodes: list[NodeType]):
        self.nodes = list(nodes)

        n_nodes = len(self.nodes)
        bottom = []
        for i in range(n_nodes):
            all_edges_row = ~bazeros(n_nodes)
            all_edges_row[i] = False
            bottom.append(fbarray(all_edges_row))
        self.bottom = tuple(bottom)

    def preprocess_data(self, data: Iterable[nx.Graph]) -> Iterator[PatternType]:
        """Convert networkx graph to GraphPS PatternType"""
        nodes_id_map = {node: i for i, node in enumerate(self.nodes)}
        n_nodes = len(nodes_id_map)

        empty_table = [bazeros(n_nodes) for _ in range(n_nodes)]
        for graph in data:
            mtrx = [empty_row.copy() for empty_row in empty_table]
            for edge in graph.edges:
                i, j = [nodes_id_map[node] for node in edge]
                mtrx[i][j] = mtrx[j][i] = True

            yield tuple([fbarray(row) for row in mtrx])

    def postprocess_data(self, data: Iterable[PatternType]) -> Iterator[nx.Graph]:
        """Convert graph from GraphPS PatternType to a networkx graph"""
        for matrix in data:
            edges = [(i, i+j) for i, row in enumerate(matrix) for j in row[i:].itersearch(True)]
            edges_verb = [(self.nodes[i], self.nodes[j]) for i, j in edges]
            yield nx.Graph(edges_verb)

    def join_patterns(self, a: PatternType, b: PatternType) -> PatternType:
        """Return the most precise common pattern, describing both patterns `a` and `b`"""
        return self._intersect_graphs(a, b)

    def iter_bin_attributes(self, data: list[PatternType])\
            -> Iterator[tuple[PatternType, fbarray]]:
        """Iterate binary attributes obtained from `data` (from the most general to the most precise ones)

        :parameter
            data: list[PatternType]
             list of object descriptions
        :return
            iterator of (description: PatternType, extent of the description: frozenbitarray)
        """
        n_objects = len(data)
        edges_extents: dict[tuple[int, int], bitarray] = dict()
        for matrix_i, matrix in enumerate(data):
            for i, row in enumerate(matrix):
                for j in row[i+1:].itersearch(True):
                    j = i+1+j
                    if (i, j) not in edges_extents:
                        edges_extents[(i, j)] = bazeros(n_objects)
                    edges_extents[(i, j)][matrix_i] = True

        full_extent = ~bazeros(n_objects)
        n_nodes = len(self.nodes)
        for graph_size in range(0, len(edges_extents)+1):
            for edges_comb in combinations(edges_extents.keys(), graph_size):
                matrix = [bazeros(n_nodes) for _ in range(n_nodes)]
                for (i, j) in edges_comb:
                    matrix[i][j] = matrix[j][i] = True

                extent = reduce(lambda a, b: a & b, (edges_extents[edge] for edge in edges_comb), full_extent)
                yield matrix, extent

        if len(edges_extents) < n_nodes * (n_nodes-1)/2:
            yield self.bottom, ~full_extent


    @staticmethod
    def _intersect_graphs(graph_a: PatternType, graph_b: PatternType) -> PatternType:
        return tuple(row_a & row_b for row_a, row_b in zip(graph_a, graph_b))

    @staticmethod
    def _is_subgraph(graph_a: PatternType, graph_b: PatternType) -> bool:
        return all(row_a & row_b == row_a for row_a, row_b in zip(graph_a, graph_b))


if __name__ == '__main__':
    ps = GraphPS('abcdefg')
    g1 = nx.Graph([('a', 'b'), ('b', 'c'), ('c', 'd')])
    g2 = nx.Graph([('a', 'b'), ('b', 'c')])
    g3 = nx.Graph([('a', 'b'), ('c', 'd')])

    data = list(ps.preprocess_data([g1, g2, g3]))

    for pattern, extent in ps.iter_bin_attributes(data):
        edges = next(ps.postprocess_data([pattern])).edges
        print(edges, extent)
