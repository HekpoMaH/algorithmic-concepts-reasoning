# algorithmic-concepts-reasoning

## Dependencies
All experiments were run with Python 3.8.5. The following packages are required to run the experiments (obtained using `pipreqs`):
```
torch_scatter==2.0.5
pandas==1.2.3
tqdm==4.49.0
torch_geometric==1.6.3
docopt==0.6.2
matplotlib==3.3.4
numpy==1.19.5
torch==1.7.0+cu110
networkx==2.4
algos==0.0.5
deep_logic==4.0.4
overrides==6.1.0
pytorch_lightning==1.3.8
schema==0.7.4
scikit_learn==0.24.2
seaborn==0.11.1
simple==0.1.1
sympy==1.8
```

## Code organisation

Currently the code is split into two folders, due to the specificity of the
implementation of Kruskal's algorithm:
1. BFS and parallel coloring heuristic (`algos`)
1. Kruskal's Minimum Spanning Tree (MST) algorithm (`algos/mst`)

### Preparing datasets

The first two algorithms listed above read their datasets from disk. Therefore,
before performing any experiments on them, run.
```
python -m algos.prepare_datasets
```

The implementation of Kruskal's doesn't require any such preparation, but
generates the datasets on every run.

In both cases, seeds are fixed, so that datasets generated are the same no
matter how many times one runs `prepare_datasets.py` or  re-generates Kruskal's
data.
