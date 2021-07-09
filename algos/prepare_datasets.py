import random
import numpy as np
import torch
import shutil
from algos.hyperparameters import get_hyperparameters
from algos.datasets import ParallelColoringDataset, ParallelColoringSingleGeneratorDataset, CombinedGeneratorsDataset, BFSSingleIterationDataset

seed = get_hyperparameters()['seed']
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
print("SEEDED")

NUM_NODES = [20]
for dataclass, rootdir in zip([CombinedGeneratorsDataset, ParallelColoringDataset], ['BFS', 'parallel_coloring']):
    rd = f'./algos/{rootdir}'
    try:
        shutil.rmtree(rd)
    except FileNotFoundError:
        pass

for num_nodes in NUM_NODES:
    for dataclass, rootdir in zip([CombinedGeneratorsDataset, ParallelColoringDataset], ['BFS', 'parallel_coloring']):
        rd = f'./algos/{rootdir}'
        def _construct(split):
            if dataclass == CombinedGeneratorsDataset:
                f = dataclass(rd, BFSSingleIterationDataset, split=split, node_degree=5, generators=get_hyperparameters()['generators'], num_nodes=num_nodes)
            else:
                f = dataclass(rd, ParallelColoringDataset, split=split, node_degree=5, generators=get_hyperparameters()['generators'], num_nodes=num_nodes)
            return f

        for split in ['train', 'val', 'test']:
            f = _construct(split)

        f0 = _construct('train')
        f1 = _construct('test')

        assert len(f0[0].edge_index[0]) != len(f1[0].edge_index[0]) or not torch.eq(f0[0].edge_index, f1[0].edge_index).all()

dataset = CombinedGeneratorsDataset('./algos/BFS', BFSSingleIterationDataset, split='test', generators=get_hyperparameters()['generators'], num_nodes=20)
dataset = ParallelColoringDataset('./algos/parallel_coloring', ParallelColoringDataset, split='test', generators=get_hyperparameters()['generators'], num_nodes=20, node_degree=5)

NUM_NODES = [50, 100]
for num_nodes in NUM_NODES:
    for dataclass, rootdir in zip([CombinedGeneratorsDataset, ParallelColoringDataset], ['BFS', 'parallel_coloring']):
        rd = f'./algos/{rootdir}'
        def _construct(split):
            if dataclass == CombinedGeneratorsDataset:
                f = dataclass(rd, BFSSingleIterationDataset, split=split, node_degree=5, generators=get_hyperparameters()['generators'], num_nodes=num_nodes)
            else:
                f = dataclass(rd, ParallelColoringDataset, split=split, node_degree=5, generators=get_hyperparameters()['generators'], num_nodes=num_nodes)
            return f

        for split in ['test']:
            f = _construct(split)

        f0 = _construct('train')
        f1 = _construct('test')

        assert len(f0[0].edge_index[0]) != len(f1[0].edge_index[0]) or not torch.eq(f0[0].edge_index, f1[0].edge_index).all()
