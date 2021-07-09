import math
import random
import numpy as np
import re
import os.path as osp
import torch
import torch.utils.data as torch_data
import torch_geometric.utils as tg_utils
from torch_geometric.data import InMemoryDataset, Data
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm
from algos.deterministic.BFS import do_BFS
from algos.deterministic.JP_coloring import jones_plassmann
from algos.hyperparameters import get_hyperparameters
from pprint import pprint

_DEVICE = get_hyperparameters()['device']
def alphanum_key(s):
    """ Turn a string into a list of string and number chunks.
        "z23a" -> ["z", 23, "a"]
    """
    def tryint(s):
        try:
            return int(s)
        except:
            return s
    return [ tryint(c) for c in re.split('([0-9]+)', s) ]

def _get_datapoints_per_split(split, train=300, non_train=50):
    if split == 'train':
        return train
    return non_train

def _sample_edge_index(kwargs, generator):
    def caveman_special(c=2,k=20,p_path=0.1,p_edge=0.3):
        p = p_path
        path_count = max(int(np.ceil(p * k)), 1)
        G = nx.caveman_graph(c, k)
        # remove 50% edges
        p = 1 - p_edge
        for (u, v) in list(G.edges()):
            if np.random.rand() < p and ((u < k and v < k) or (u >= k and v >= k)):
                G.remove_edge(u, v)
        # add path_count links
        for i in range(path_count):
            [cluster1, cluster2] = np.random.choice(c, 2, replace=False)
            u = np.random.randint(cluster1 * k, (cluster1 + 1) * k)
            v = np.random.randint(cluster2 * k, (cluster2 + 1) * k)
            G.add_edge(u, v)
        return tg_utils.from_networkx(G).edge_index
    def closest_factor_to_sqrt(n):
        closest = 1
        for i in range(1, n):
            if i*i >= n:
                break
            if n % i == 0:
                closest = i
        return closest

    # we generate a variety of graph input structures to avoid overfitting
    # and promote generalisation
    # kwargs = self.kwargs
    if generator == 'ER':
        # Erdos-Renyi graph
        edge_prob = kwargs.get('edge_prob', min(math.log2(kwargs['num_nodes'])/kwargs['num_nodes'], 0.5))
        edge_index = tg_utils.erdos_renyi_graph(kwargs['num_nodes'], edge_prob, directed=False)

    elif generator == 'ladder':
        # Ladder (2x(N/2)) graph
        nxg = nx.generators.ladder_graph(kwargs['num_nodes'] // 2)
        edge_index = tg_utils.from_networkx(nxg).edge_index
        perm = torch.randperm(kwargs['num_nodes'])
        for i in range(len(edge_index)):
            for j in range(len(edge_index[i])):
                edge_index[i][j] = perm[edge_index[i][j]]


    elif generator == 'grid':
        # Grid graph (generates closest to a square grid)
        cf = closest_factor_to_sqrt(kwargs['num_nodes'])
        nxg = nx.generators.grid_2d_graph(cf, kwargs['num_nodes'] // cf)
        edge_index = tg_utils.from_networkx(nxg).edge_index
        perm = torch.randperm(kwargs['num_nodes'])
        for i in range(len(edge_index)):
            for j in range(len(edge_index[i])):
                edge_index[i][j] = perm[edge_index[i][j]]

    elif generator == 'tree':
        # Tree graph from the Prufer sequence
        seq = torch.randint(0, kwargs['num_nodes'], (kwargs['num_nodes']-2, )).tolist()
        nxg = nx.algorithms.tree.coding.from_prufer_sequence(seq)
        edge_index = tg_utils.from_networkx(nxg).edge_index

    elif generator == 'BA':
        # Barabasi-Albert preferential attachment with 4 or 5 attachments per new node
        seq = torch.randperm(kwargs['num_nodes'])[:-1]
        nedg = 4 if np.random.randint(1, size=1) == 1 else 5
        edge_index = tg_utils.barabasi_albert_graph(kwargs['num_nodes'], nedg)

    elif generator == 'disconnected':
        # Graph that consists of two disconnected halfs
        edge_prob = kwargs.get('edge_prob', min(math.log2(kwargs['num_nodes'])/kwargs['num_nodes'], 0.2))-0.05
        assert edge_prob >= 0
        edge_index1 = tg_utils.erdos_renyi_graph(kwargs['num_nodes']//2, edge_prob, directed=False)
        edge_index2 = tg_utils.erdos_renyi_graph(kwargs['num_nodes']//2, edge_prob, directed=False) + kwargs['num_nodes']//2
        edge_index = torch.cat((edge_index1, edge_index2), dim=-1)

    elif generator == 'caveman':
        edge_index = caveman_special(c=4, k=kwargs['num_nodes'] // 4)

    elif generator == '4-community':
        szs = [kwargs['num_nodes'] // 4, kwargs['num_nodes'] // 4, kwargs['num_nodes'] // 4,kwargs['num_nodes'] // 4 + kwargs['num_nodes'] % 4]
        sum_sz = 0
        edge_index = [[], []]
        for size in szs:
            ei = tg_utils.erdos_renyi_graph(size, 0.7, directed=False)
            for iei in range(2):
                edge_index[iei].extend((ei[iei]+sum_sz).tolist())
            sum_sz += size

        for n1 in range(kwargs['num_nodes']):
            for n2 in range(kwargs['num_nodes']):
                if abs(n1-n2) > kwargs['num_nodes'] // 4 and torch.rand((1))[0] < 0.01:
                    edge_index[0].append(n1)
                    edge_index[1].append(n2)
                    edge_index[1].append(n1)
                    edge_index[0].append(n2)

        edge_index = torch.tensor(edge_index)

    elif generator == 'deg5':
        node_degrees = [5 for _ in range(kwargs['num_nodes'])]
        graph = nx.generators.degree_seq.configuration_model(node_degrees, create_using=nx.Graph)
        edge_index = tg_utils.from_networkx(graph).edge_index


    # We additionally insert a self-edge to every node in the graphs, in
    # order to support easier retention of self-information through message
    # passing.
    edge_index, _ = tg_utils.add_remaining_self_loops(edge_index, num_nodes=kwargs['num_nodes'])
    return edge_index

class GraphDatasetBase(InMemoryDataset):
    def __init__(self, root, device='cuda', split='train', transform=None, pre_transform=None):
        self.device = device
        assert split in ['train', 'val'] or 'test' in split

        self.split = split
        super(GraphDatasetBase, self).__init__(root, transform, pre_transform)

    @property
    def processed_file_names(self):
        # Raw dir below is not a typo
        return [osp.join(self.raw_dir, self.split, 'data.pt')]

class _CustomInMemoryDataset(InMemoryDataset):
    def __init__(self,
                 root,
                 transform=None,
                 pre_transform=None,
                 datapoints_train=300,
                 datapoints_non_train=50):
        self.datapoints_train = datapoints_train
        self.datapoints_non_train = datapoints_non_train
        super(_CustomInMemoryDataset, self).__init__(root, transform, pre_transform)


    @property
    def processed_file_names(self):
        return ['data.pt']

    def process(self):
        data_list = [self._sample_datapoint() for _ in tqdm(range(_get_datapoints_per_split(self.split, train=self.datapoints_train, non_train=self.datapoints_non_train)))]
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        print("SAVING AT", self.processed_paths[0])


class BFSSingleIterationDataset(_CustomInMemoryDataset):
    '''
    The class takes as input what generator/split to generate e.g. ER for
    testing 2x sized graphs (test_2x), together with how many points to
    generate for training and non-training splits. Additional parameters
    can be passed, but please refer to the code for them.

    The class returns a PyG dataset where each Data object has a 'temporal'
    dimension with # of iterations = # objects.
    '''

    def __init__(self,
                 root,
                 split='train',
                 generator='ER',
                 transform=None,
                 pre_transform=None,
                 datapoints_train=300,
                 datapoints_non_train=50,
                 **kwargs):
        assert split in ['train', 'val'] or 'test' in split
        self.split = split
        self.kwargs = kwargs
        self.generator = generator
        self.datapoints_train = datapoints_train
        self.datapoints_non_train = datapoints_non_train
        super(BFSSingleIterationDataset, self).__init__(root, transform, pre_transform, datapoints_train, datapoints_non_train)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_dir(self):
        return osp.join(self.root, str(self.generator), str(self.kwargs['num_nodes']), 'processed', self.split)

    def _sample_datapoint(self):
        edge_index = _sample_edge_index(self.kwargs, self.generator)
        start_node = torch.randint(self.kwargs['num_nodes'], (1,)).item()
        datapoint = do_BFS(self.kwargs['num_nodes'], edge_index, start_node=start_node)
        assert tg_utils.is_undirected(edge_index)
        return datapoint

    def process(self):
        super(BFSSingleIterationDataset, self).process()

class CombinedGeneratorsDataset(_CustomInMemoryDataset):
    def __init__(self,
                 root,
                 inside_class,
                 split='train',
                 generators=['ER'],
                 transform=None,
                 pre_transform=None,
                 datapoints_train_per_generator=100,
                 datapoints_non_train_per_generator=10,
                 **kwargs):
        self.split = split
        self.inside_class = inside_class
        self.generators = generators
        self.kwargs = kwargs
        self.transform = transform
        self.pre_transform = pre_transform
        self.datapoints_train_per_generator = datapoints_train_per_generator
        self.datapoints_non_train_per_generator = datapoints_non_train_per_generator
        super(CombinedGeneratorsDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
        positives = self.data.concepts.flatten(start_dim=0, end_dim=1).float().sum(0)
        negatives = len(self.data.concepts.flatten(start_dim=0, end_dim=1)) - positives
        self.concept_pos_weights = negatives/positives
        self.concept_pos_weights = self.concept_pos_weights.to(_DEVICE)
        positives = self.data.termination.flatten(start_dim=0, end_dim=1).float().sum(0)
        negatives = len(self.data.termination.flatten(start_dim=0, end_dim=1)) - positives
        self.termination_pos_weights = negatives/positives
        self.termination_pos_weights = self.termination_pos_weights.to(_DEVICE)

    @property
    def processed_dir(self):
        return osp.join(self.root, '-'.join(self.generators), str(self.kwargs['num_nodes']), 'processed', self.split)

    def process(self):
        data_list = []
        for generator in self.generators:
            inside_dataset = self.inside_class(
                self.root,
                split=self.split,
                generator=generator,
                transform=self.transform,
                pre_transform=self.pre_transform,
                datapoints_train=self.datapoints_train_per_generator,
                datapoints_non_train=self.datapoints_non_train_per_generator,
                **self.kwargs)
            dataset = [
                el for el in inside_dataset
            ]
            data_list.extend(dataset)


        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        print("SAVING AT", self.processed_paths[0])

class ParallelColoringDataset(_CustomInMemoryDataset):
    '''
    A class for the parallel coloring task. In terms of return values
    it is similar to the BFS ones, but it has a different generation
    technique: It generates random graphs with degree of node_degree.

    From networkx:
    If the maximum degree d_m in the sequence is O(m^{1/4}) then the algorithm
    produces almost uniform random graphs in O(m*d_m) time where m is the number
    of edges.


    Moshen Bayati, Jeong Han Kim, and Amin Saberi, A sequential algorithm for
    generating random graphs. Algorithmica, Volume 58, Number 4, 860-910, DOI:
        10.1007/s00453-009-9340-1
    '''

    def __init__(self,
                 root,
                 inside_class, #NOTE Ignored, so as to match constructor of other parallel coloring class
                 split='train',
                 node_degree=5,
                 transform=None,
                 pre_transform=None,
                 datapoints_train=800,
                 datapoints_non_train=80,
                 **kwargs):
        assert split in ['train', 'val'] or 'test' in split
        self.split = split
        self.kwargs = kwargs
        self.node_degree = node_degree
        self.datapoints_non_train = datapoints_non_train
        super(ParallelColoringDataset, self).__init__(root, transform, pre_transform, datapoints_train, datapoints_non_train)
        self.data, self.slices = torch.load(self.processed_paths[0])
        _, counts = torch.unique(torch.tensor(self.data.y), return_counts=True)
        self.class_weights = torch.tensor([sum(counts)/(c * 6) for c in counts]).to(_DEVICE)
        positives = self.data.concepts.flatten(start_dim=0, end_dim=1).float().sum(0)
        negatives = len(self.data.concepts.flatten(start_dim=0, end_dim=1)) - positives
        self.concept_pos_weights = negatives/positives
        self.concept_pos_weights = self.concept_pos_weights.to(_DEVICE)

        positives = self.data.termination.flatten(start_dim=0, end_dim=1).float().sum(0)
        negatives = len(self.data.termination.flatten(start_dim=0, end_dim=1)) - positives
        self.termination_pos_weights = negatives/positives
        self.termination_pos_weights = self.termination_pos_weights.to(_DEVICE)

    @property
    def processed_dir(self):
        return osp.join(self.root, str(self.node_degree), str(self.kwargs['num_nodes']), 'processed', self.split, '' if self.datapoints_non_train == 80 else str(self.datapoints_non_train))

    def _sample_datapoint(self):
        node_degrees = [self.node_degree for _ in range(self.kwargs['num_nodes'])]
        graph = nx.generators.degree_seq.configuration_model(node_degrees, create_using=nx.Graph)
        graph.remove_edges_from(nx.selfloop_edges(graph))
        assert (np.array([deg for _, deg in graph.degree(graph.nodes)]) <= self.node_degree).all()
        data = None
        while data is None:
            data = jones_plassmann(graph)
        data.edge_index, _ = tg_utils.add_remaining_self_loops(data.edge_index)

        return data

    def process(self):
        super(ParallelColoringDataset, self).process()

class ParallelColoringSingleGeneratorDataset(_CustomInMemoryDataset):

    def __init__(self,
                 root,
                 split='train',
                 node_degree=5,
                 generator='ER',
                 transform=None,
                 pre_transform=None,
                 datapoints_train=100,
                 datapoints_non_train=10,
                 **kwargs):
        assert split in ['train', 'val'] or 'test' in split
        self.split = split
        self.node_degree = node_degree
        self.kwargs = kwargs
        self.generator = generator
        super(ParallelColoringSingleGeneratorDataset, self).__init__(root, transform, pre_transform, datapoints_train, datapoints_non_train)
        self.data, self.slices = torch.load(self.processed_paths[0])
        _, counts = torch.unique(torch.tensor(self.data.y), return_counts=True)
        self.class_weights = torch.tensor([sum(counts)/(c * 6) for c in counts]).to(_DEVICE)
        positives = self.data.concepts.flatten(start_dim=0, end_dim=1).float().sum(0)
        negatives = len(self.data.concepts.flatten(start_dim=0, end_dim=1)) - positives
        self.concept_pos_weights = negatives/positives
        self.concept_pos_weights = self.concept_pos_weights.to(_DEVICE)

        positives = self.data.termination.flatten(start_dim=0, end_dim=1).float().sum(0)
        negatives = len(self.data.termination.flatten(start_dim=0, end_dim=1)) - positives
        self.termination_pos_weights = negatives/positives
        self.termination_pos_weights = self.termination_pos_weights.to(_DEVICE)

    @property
    def processed_dir(self):
        return osp.join(self.root, str(self.generator), str(self.kwargs['num_nodes']), 'processed', self.split)

    @staticmethod
    def _limit_degree(edge_index, max_degree=5):
        new_edge_index = [[], []]
        deg_count = {}
        for u, v in zip(edge_index[0].tolist(), edge_index[1].tolist()):
            if u not in deg_count:
                deg_count[u] = 0

            if v not in deg_count:
                deg_count[v] = 0

            if deg_count[u] >= max_degree or deg_count[v] >= max_degree:
                continue
            new_edge_index[0].append(u)
            new_edge_index[1].append(v)
            new_edge_index[1].append(u)
            new_edge_index[0].append(v)
            deg_count[u] += 1
            deg_count[v] += 1

        return torch.tensor(new_edge_index)

    def _sample_datapoint(self):
        def _sample():
            edge_index = _sample_edge_index(self.kwargs, self.generator)
            edge_index, _ = tg_utils.remove_self_loops(edge_index)
            edge_index = ParallelColoringSingleGeneratorDataset._limit_degree(edge_index, max_degree=4 if self.generator == '4-community' else 5)
            graph = tg_utils.to_networkx(Data(edge_index=edge_index, num_nodes=self.kwargs['num_nodes']), to_undirected=True)
            graph.remove_edges_from(nx.selfloop_edges(graph))
            return edge_index, graph
        edge_index, graph = _sample()
        assert (np.array([deg for _, deg in graph.degree(graph.nodes)]) <= self.node_degree).all(), np.array([deg for _, deg in graph.degree(graph.nodes)])
        data = None
        while data is None:
            data = jones_plassmann(graph)
        data.edge_index, _ = tg_utils.add_remaining_self_loops(data.edge_index)

        return data

    def process(self):
        super(ParallelColoringSingleGeneratorDataset, self).process()

class TerminationCombinationDataset: #NOTE This is NOT a pytorch dataset
    def __init__(self, existing_dataset):
        self.existing_dataset = existing_dataset
        self.dataset = []
        self.seen = set()
        for p in self.existing_dataset:
            for timestep in range(p.concepts.shape[1]):
                concepts = torch.unique(p.concepts[:, timestep+1], dim=0)
                tpl = tuple(tuple(t) for t in concepts.tolist())
                if tpl not in self.seen:
                    self.seen.add(tpl)
                    self.dataset.append((concepts.numpy(), p.termination[:, timestep].numpy()))
                if not p.termination[:, timestep].all():
                    break
        print(p.x.shape)
        print(p.concepts.shape)


if __name__ == '__main__':
    for spl in ['train', 'val', 'test']:
        f = ParallelColoringDataset('./algos/parallel_coloring', None, split=spl, num_nodes=20, generators=['ER'])
        t = TerminationCombinationDataset(f)
        print(type(t.dataset))
        print(spl, len(t.dataset), len(t.seen))
