import torch
import math
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import torch_geometric.utils as tg_utils
import torch_geometric


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

        edge_index = torch.tensor(edge_index).long()

    elif generator == 'deg5':
        node_degrees = [5 for _ in range(kwargs['num_nodes'])]
        graph = nx.generators.degree_seq.configuration_model(node_degrees, create_using=nx.Graph)
        edge_index = tg_utils.from_networkx(graph).edge_index

    if len(edge_index[0]):
        edge_index = tg_utils.to_undirected(edge_index, num_nodes=torch.tensor(kwargs['num_nodes']).long())
    return edge_index 

def generate_random_graph(n_nodes: int = 4, random_state: int = 42, generator='ER'):
    """
    Generate a random graph.

    :param n_nodes: number of graph nodes.
    :param random_state: random state.
    :return: a random graph.
    """
    kwargs = {
        'num_nodes': n_nodes,
    }
    data = torch_geometric.data.Data(x=torch.zeros((n_nodes, 1)), edge_index=_sample_edge_index(kwargs, generator))
    G = tg_utils.to_networkx(data, to_undirected=True)
    while not nx.is_connected(G):
        data = torch_geometric.data.Data(x=torch.zeros((n_nodes, 1)),edge_index=_sample_edge_index(kwargs, generator))
        G = tg_utils.to_networkx(data, to_undirected=True)

    np.random.seed(random_state)
    random_edge_weights = (torch.randperm(1023)+1)[:len(G.edges)]
    for ((u,v), weight) in zip(G.edges, random_edge_weights):
        G.edges[u,v]['weight'] = weight
    return G


def array_to_bits(matrix: np.array, bits: int, n_ints = None):
    """
    Transform an array of numbers from decimal representation to binary.

    Reference: Yan, Yujun, et al. "Neural execution engines: Learning to execute subroutines." arXiv preprint arXiv:2006.08084 (2020).
    https://arxiv.org/pdf/2006.08084.pdf
    Page 4, Bitwise Embeddings

    :param matrix: array of integers
    :param bits: number of bits for the binary representation.
    :param n_ints: number of decimal places to consider for floating point numbers.
    :return: an tensor of numbers on binary.
    """
    if n_ints is not None:
        matrix = (matrix * n_ints).astype(int)
    n_samples, n_features = matrix.shape
    get_bin = lambda n, z: np.array(list(format(n, 'b').zfill(z))).astype(int)
    x_train_bits = np.stack([get_bin(num, bits) for num_list in matrix for num in num_list])
    return torch.FloatTensor(x_train_bits).view(n_samples, n_features, bits)


if __name__ == '__main__':
    # generate random graph
    G = generate_random_graph(n_nodes=5, random_state=1)
    edge_weights_dict = nx.get_edge_attributes(G, 'weight')
    pos = nx.spring_layout(G)

    plt.figure()
    nx.draw(G, pos=pos, with_labels=True)
    nx.draw_networkx_edge_labels(G, pos=pos, edge_labels=edge_weights_dict)
    plt.show()

    # array to bits
    adjacency = np.array(nx.to_numpy_matrix(G))
    edge_features = np.array([v for k, v in nx.get_edge_attributes(G, "weight").items()])[[np.newaxis]]
    print(f'Edge features: {edge_features}')
    array_to_bits(edge_features, 8)
