import torch
import numpy as np
from torch_geometric.data import Data
import networkx as nx

from algos.mst.data_structure import QueryUnionFind
from algos.mst.utils import array_to_bits, generate_random_graph


class MinimumSpanningTree:

    def __init__(self, G):
        self.n_nodes = len(G.nodes)
        self.n_edges = len(G.edges)
        self.max_time = len(G.edges) + 1
        self.time = np.arange(self.max_time)

        # sort edges by weight (descending)
        self.edge_features = np.array([v for k, v in nx.get_edge_attributes(G, "weight").items()])
        self.edge_order = np.argsort(self.edge_features)
        self.edge_nodes = np.array(G.edges)

        # data structure for disjoint-set unions operations required for the minimum spanning tree
        self.subtrees = QueryUnionFind(self.n_nodes)

        # node features
        self.node_features = np.zeros((self.max_time, self.n_nodes, 2))

        # edge concepts
        self.lighter_edges_visited = np.zeros(self.n_edges)
        self.nodes_in_same_set = np.zeros(self.n_edges)
        self.edge_in_mst = np.zeros(self.n_edges)
        self.edge_concepts = np.zeros((self.max_time, self.n_edges, 3))

        # targets
        # edges
        self.edge_is_visited = np.zeros(self.n_edges)
        self.edge_targets = np.zeros((self.max_time, self.n_edges, 1))
        # nodes (PGN)
        self.nodes_in_different_sets = np.zeros(self.max_time)
        self.nodes_not_in_path_to_roots = np.zeros((self.max_time, self.n_nodes))
        self.asymmetric_pointers = np.zeros((self.max_time, self.n_nodes))
        self.asymmetric_pointers[0] = np.arange(self.n_nodes)
        self.edge_index = np.zeros((self.max_time, self.n_nodes, 2))

        # termination
        self.termination = np.zeros(self.max_time)

    def run(self, debug=False):
        # initialize the first time step (it'll be almost empty)
        uv = np.zeros(self.n_nodes)
        self.nodes_in_different_sets[0] = 1
        self.node_features[0] = np.array([self.subtrees.r, uv]).T
        self.edge_index[0] = np.array([[s, t] for s, t in self.subtrees.pi.items()])
        if debug: print(self.time)
        for i, time in enumerate(self.time[1:]):
            if debug: print(f'\nTime: {time} - i: {i}')

            # at each time step, pick the edge with lowest weight
            # mark the edge as visited
            self.lighter_edges_visited[self.edge_order[i]] = 1
            # save current concept states for each edge
            niss = []
            for u,v in self.edge_nodes:
                niss.append(self.subtrees.safe_query(u, v))

            u, v = self.edge_nodes[self.edge_order][i]
            self.edge_concepts[time] = np.array([self.lighter_edges_visited,
                                                 niss,
                                                 self.edge_in_mst]).T

            # mark the nodes corresponding to the current edge
            uv = np.zeros(self.n_nodes)
            uv[u] = uv[v] = 1
            self.node_features[time] = np.array([self.subtrees.r, uv]).T

            # Check if the current nodes are in the same set.
            # If they were not, then the two sets will be unified and
            # the current edge will be in the minimum spanning tree
            query_union_out = self.subtrees.query_union(u, v)
            self.edge_in_mst[self.edge_order[i]] = query_union_out

            # self.subtrees.query_union(u, v),
            # This is the target for each node:
            # if the current edge is in the MST, then the current nodes were in different sets before!
            self.nodes_in_different_sets[time] = query_union_out
            # mark the pointers which have changed due to path compression
            self.nodes_not_in_path_to_roots[time] = self.subtrees.mu
            # save asymmetric pointers
            self.asymmetric_pointers[time] = np.array(list(self.subtrees.pi.values()))
            self.edge_index[time] = np.array([[s, t] for s, t in self.subtrees.pi.items()])

            # mark the current edge as visited
            self.edge_is_visited[i] = 1
            # save current target states for each edge
            self.edge_targets[time] = np.array([self.edge_in_mst]).T

            # the nodes attached to the current edge are now in the same for sure!
            self.nodes_in_same_set[i] = 1

            if debug:
                print(f'Edge nodes:\n{self.edge_nodes.T}')
                print(f'Edge feature: {self.edge_features}')
                print(f'Current edge: ({u}, {v})')
                print(f'Node features:\n{self.node_features[time]}')
                print(f'Edge concepts:\n{self.edge_concepts[time]}')
                print(f'Edge targets:\n{self.edge_targets[time]}')
                print(f'Nodes in different sets: {self.nodes_in_different_sets}')
                print(f'Nodes not in path to roots: {self.nodes_not_in_path_to_roots[time]}')
                print(f'Asymmetric pointers:\n{self.asymmetric_pointers[time]}')
                print(f'Termination: {self.termination}')

            self.termination[-1] = 1

    def to_data(self, max_time=10, bits=16):
        """
        Padding arrays and return nice tensors.
        For node tensors:
            1st dimension: number of nodes
            2nd dimension: number of time steps
            3rd dimension: others
        For edge tensors:
            1st dimension: 1 (batch dimension)
            2nd dimension: number of edges
            3rd dimension: number of time steps
            4th dimension: others

        :param max_time: pad time/edge dimensions tensors up to this value.
        :param bits: number of bits for bitwise encoding.
        :param debug: debug flag.
        :return: data tensors.
        """
        self.edge_index = self.edge_index.transpose(2, 0, 1)
        self.edge_index = pad_time(self.edge_index,
                                   (self.edge_index.shape[0], max_time,
                                    self.edge_index.shape[2]))

        def stupid_stupid_pad(pi, max_time):
            out = torch.full((max_time, pi.shape[1]), -1)
            out[:pi.shape[0]] = pi
            return out


        apout = stupid_stupid_pad(torch.tensor(self.asymmetric_pointers), max_time)
        assert (apout[:self.asymmetric_pointers.shape[0]] == torch.tensor(self.asymmetric_pointers)).all()

        self.edge_concepts = self.edge_concepts.transpose(1, 0, 2)

        self.edge_concepts = pad_time(self.edge_concepts,
                                      (self.edge_concepts.shape[0], max_time, self.edge_concepts.shape[2]))
        self.edge_concepts = pad_edge_concepts(self.edge_concepts, max_time)[np.newaxis]

        self.edge_targets = self.edge_targets.transpose(1, 0, 2)
        self.edge_targets = pad_time(self.edge_targets,
                                     (self.edge_targets.shape[0], max_time, self.edge_targets.shape[2]))
        self.edge_targets = pad_edge_concepts(self.edge_targets, max_time)[np.newaxis]

        edge_features_binary = array_to_bits(self.edge_features[np.newaxis], bits=bits).squeeze(0)
        edge_features_binary = pad_edges(edge_features_binary, max_time)[np.newaxis, :, np.newaxis]
        self.edge_features = self.edge_features.argsort()
        self.edge_features = pad_edge_features_int(self.edge_features, max_time)[np.newaxis, np.newaxis, :, np.newaxis]

        self.node_features = self.node_features.transpose(1, 0, 2)
        self.node_features = pad_node_features(self.node_features, max_time)

        mzero = np.zeros(max_time)
        mzero[:self.nodes_in_different_sets.shape[0]] = self.nodes_in_different_sets
        self.nodes_in_different_sets = mzero[np.newaxis, np.newaxis]

        self.nodes_not_in_path_to_roots = self.nodes_not_in_path_to_roots[:, :, np.newaxis]
        self.nodes_not_in_path_to_roots = self.nodes_not_in_path_to_roots.transpose(1, 0, 2)
        self.nodes_not_in_path_to_roots = pad_time(self.nodes_not_in_path_to_roots,
                                                   (self.nodes_not_in_path_to_roots.shape[0], max_time,
                                                    self.nodes_not_in_path_to_roots.shape[2]))

        mone = np.ones(max_time)
        mone[:self.termination.shape[0]] = self.termination
        self.termination = mone[np.newaxis, np.newaxis]

        self.time = np.arange(max_time)[np.newaxis, np.newaxis]

        data = Data(
            asymmetric_pointers_index=apout,
            edge_concepts=torch.FloatTensor(self.edge_concepts),
            edge_features=torch.FloatTensor(self.edge_features),
            edge_features_binary=torch.FloatTensor(edge_features_binary),
            edge_index=torch.LongTensor(self.edge_index),
            edge_nodes_index=torch.LongTensor(self.edge_nodes.T),
            edge_targets=torch.FloatTensor(self.edge_targets),
            x=torch.FloatTensor(self.node_features),
            nodes_in_different_sets=torch.FloatTensor(self.nodes_in_different_sets),
            nodes_not_in_path_to_roots=torch.FloatTensor(self.nodes_not_in_path_to_roots),
            termination=torch.FloatTensor(self.termination),
            time=torch.FloatTensor(self.time),
        )

        return data


def pad_time(matrix, shape_pad):
    matrix_pad = np.tile(matrix[:, -1][:, np.newaxis], reps=shape_pad[1]).reshape(shape_pad)
    matrix_pad[:matrix.shape[0], :matrix.shape[1]] = matrix
    return matrix_pad


def pad_node_features(matrix, max_time):
    matrix_pad = np.zeros((matrix.shape[0], max_time, matrix.shape[2]))
    matrix_pad[:matrix.shape[0], :matrix.shape[1], :matrix.shape[2]] = matrix
    return matrix_pad


def pad_Pi(matrix, shape_pad):
    matrix_pad = np.tile(matrix[:, -1][:, np.newaxis], reps=shape_pad[1]).reshape(shape_pad)
    matrix_pad[:matrix.shape[0], :matrix.shape[1]] = matrix
    return matrix_pad


def pad_edges(matrix, max_edges):
    matrix_pad = np.zeros((max_edges, matrix.shape[1]))
    matrix_pad[:matrix.shape[0], :matrix.shape[1]] = matrix
    return matrix_pad


def pad_edge_features_int(matrix, max_time):
    matrix_pad = np.zeros((max_time)) - 1
    matrix_pad[:matrix.shape[0]] = matrix
    return matrix_pad


def pad_edge_nodes(matrix, max_time):
    matrix_pad = np.zeros((max_time, matrix.shape[1]))
    matrix_pad[:matrix.shape[0], :matrix.shape[1]] = matrix
    return matrix_pad


def pad_edge_concepts(matrix, max_time):
    matrix_pad = np.zeros((max_time, matrix.shape[1], matrix.shape[2])) - 1
    matrix_pad[:matrix.shape[0]] = matrix
    return matrix_pad
