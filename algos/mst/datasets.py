import numpy as np

from algos.mst.deterministic import MinimumSpanningTree
from algos.mst.utils import generate_random_graph


def generate_dataset(n_graphs: int = 30, batch_size: int = 10, n_nodes: int = 5, bits: int = 16):
    """
    Generate a dataset of random graphs having the same number of nodes.

    :param n_graphs: number of graphs to be generated.
    :param batch_size: number of graphs per batch.
    :param n_nodes: number of nodes per graph.
    :param bits: number of bits for bitwise encoded features.
    :return: a pytorch data loader.
    """

    # generate random graphs and remember the graph having most edges
    # (this will correspond to the maximum number of time steps for the algorithm)
    graph_list = []
    max_n_edges = 0
    for i in range(n_graphs):
        db4 = n_graphs // 4
        if i < db4:
            generator = 'ER'
        if i >= db4 and i < 2*db4:
            generator = 'ladder'
        if i >= 2*db4 and i < 3*db4:
            generator = 'grid'
        if i >= 3*db4:
            generator = 'BA'
        np.random.seed(i)   # for reproducibility
        n_nodes = np.random.randint(n_nodes, n_nodes + 1)
        np.random.seed(i)   # for reproducibility
        random_state = np.random.randint(10000)
        G = generate_random_graph(n_nodes, random_state, generator=generator)

        graph_list.append(G)
        if max_n_edges < len(G.edges):
            max_n_edges = len(G.edges)

    # run the (deterministic) Kruskal algorithm to find the minimum spanning tree,
    # compute a few key concepts and return them as a nice pytorch Data object
    data_list = []
    for j, G in enumerate(graph_list):
        mst = MinimumSpanningTree(G)
        mst.run(debug=False)
        data = mst.to_data(max_time=max_n_edges + 1, bits=bits)
        data_list.append(data)

    return data_list, max_n_edges, None
