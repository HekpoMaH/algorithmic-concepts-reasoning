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

Some of our scripts use Unix shell (`bash`) commands. All scripts have been tested under Ubuntu 20.04

## Code organisation

Currently the code is split into two folders, due to the specificity of the
implementation of Kruskal's algorithm:
1. BFS and parallel coloring heuristic (`algos`)
1. Kruskal's Minimum Spanning Tree (MST) algorithm (`algos/mst`)

### Preparing datasets

The first two algorithms listed above read their datasets from disk. Therefore,
before performing any experiments on them, run:
```
python -m algos.prepare_datasets
```
This may take a while, so grab a cup of coffee/tea. Of course, you can attempt
to train directly (see below), but you may end up with different datasets than
us.

The implementation of Kruskal's doesn't require any such preparation, but
generates the datasets on every run.

In both cases, seeds are fixed, so that datasets generated are the same no
matter how many times one runs `prepare_datasets.py` or  re-generates Kruskal's
data.

## Training & Testing

As we heavily relied on the `docopt` package the documentation to our script is
also a CLI interface. We **heavily** suggest checking the flags' documentation
to the scripts, but to save you some hassle, we provide examples of
configurations we used later below.

### BFS and parallel coloring

The scripts for BFS and parallel coloring are in the `algos` folder.

Training ***one*** seed is achieved via the `train.py`. `train.py` serialises
the weights of the model every 10 epochs in the `algos/serialised_models`
directory, which is automatically created the first time you attempt to train
a model. The script also saves the weights of the best model (out of all
epochs) in that folder. To have different filenames for all these
serialisations, the provided model name (as a flag to `train.py`) is modified
to `test_<model-name>_epoch_<EPOCH>.pt` and to `best_<model-name>.pt`.

If I now want to test a specific serialisation, I need to provide the full
model path (with `algos/serialised_models/`). If one wants to test the best
model, it can do so by (e.g.):
```
python -m algos.test --model-path algos/serialised_models/best_model.pt
```
*NOTE* Please, bear in mind that you need to take care of providing a `--has-GRU` flag,
if your model used a GRU gate on the update step (_automatically used_ if you
do not use teacher forcing)

If you want to test the model on every epoch serialised, e.g. for (re)producing plots,
run:
```
python -m algos.test_per_epoch --model-name model
```
This will produce a `.csv` file in the `algos/results` folder. (Created automatically)

Now, if you want to generate statistics over several seeds (e.g. for standard deviation), use
the `*several_seeds.py` scripts and `test.py`.

For training ***several*** seeds, use the `train_several_seeds.py` script. Most flags
should match there and, behind the curtains, `train_several_seeds.py` spawns
several `train.py` processes via a shell script.

If you want to test the best models across these
several seeds (e.g. for tabulating results), use:
```
python -m algos.test --model-path algos/serialised_models/best_model --use-seeds --num-seeds <NS>
```
*NOTE* Please observe we did not provide the full path, but we stopped before
`_seeds_<NUM>.pt` part of the filename.

If you want to do all tests of our tables at once, use the `--all-num-nodes`
flag and you will get nice tabular LaTeX formatting.

For generating the statistics per epoch for all the seeds use
`test_several_seeds.py` script. Behind the curtains this spawns several
`test_per_epoch.py` scripts.

#### Kruskal's algorithm

The code for Kruskal's algorithm resides in the `algos/mst` folder. Although
the implementation is different, we aimed to make the training/testing API as
close as possible --- training/testing one seed is done via
`algos/mst/train.py`, `algos/mst/test.py` and `algos/mst/test_per_epoch.py`,
respectively, training several seeds is done via
`algos/mst/train_several_seeds.py`, `algos/mst/test.py` (with the `--use-seeds`
flag) and `algos/mst/test_several_seeds.py`.

### In case you want to experiment...

And want to make modifications to our code, the rest of the repository consists of:

1. `algos/models` folder contains implementations of the *whole* architectures
   for the processor (`algorithm_processor.py`) and algorithms
   (`algorithm_base.py` for BFS, `algorithm_coloring.py` for parallel coloring,
   which reuses parts of the BFS implementation).

1. `algos/layers` folder contains *layers* of the architecture, such as
   a single MPNN layer or a PrediNet pooling, various encoders for inputs, e.g.
   bit encoders,

1. `algos/deterministic` folder contains the deterministic implementations of
   the algorithms that we use to generate our data and `algos/datasets.py`
   holds the corresponding PyTorch dataset wrappers.

1. `algos/utils.py` holds various utility functions, such as data preparation,
   iterating over a batch, adding explanations to trained models, etc.

Finally, `algos/mst` is organised in the same way, with the difference that
`algos/mst/data_structure.py` holds the deterministic implementation of
a union-find data structure.

### Examples

As this is a lot of information, we provide some examples to get you started:

For BFS run:
```
python -m algos.train_several_seeds --use-TF --use-GRU --algos BFS --epochs 500 --no-patience --model-name BFS
```
and if you want to remove concept bottlenecks and supervision add:
```
python -m algos.train_several_seeds --use-TF --use-GRU --algos BFS --epochs 500 --no-patience --model-name BFS --no-use-concepts --no-use-concepts-sv
```

For parallel coloring, you just need to change the `--algos` flag to
`parallel_coloring` and adjust hyperparameters:
```
python -m algos.train_several_seeds --use-TF --use-GRU --algos parallel_coloring --epochs 3000 --prune-epoch 2000 --L1-loss --no-patience --model-name parallel_coloring
```

And finally, for Kruskal's:
```
python -m algos.mst.train_several_seeds --epochs 100 --model-name kruskal --num-nodes 8
```

## Licence

Licensed under the Apache License, Version 2.0 (the "License"); you may
not use this file except in compliance with the License. You may obtain
a copy of the License at: http://www.apache.org/licenses/LICENSE-2.0.

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

See the License for the specific language governing permissions and
limitations under the License.