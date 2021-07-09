"""
Usage:
    plot_acc_distribution.py (--model-path=MP) [--algos=ALGO]... [options] 

Options:
    -h --help           Show this screen.

    --algos ALGO        Which algorithms to load {BFS, parallel_coloring}.
                        Repeatable parameter. [default: BFS]

    --has-GRU           Does the processor utilise GRU unit? [default: False]

    --pooling=PL           What graph pooling mechanism to use for termination.
                           One of {attention, predinet, max, mean}. [default: predinet]

    --model-path MP     Path of the model to load
    
    --num-nodes NN      Number of nodes in the graphs to test on [default: 20]

    --use-seeds         Use seeds for STD. It will automatically modify name as
                        appropriate. [default: False]

    --num-seeds NS      How many different seeds to plot. [default: 5]

    --no-next-step-pool    Do NOT use next step information for termination.
                           Use current instead. [default: False]

    --pruned               Is concept decoder pruned? [default: False]
"""
import numpy as np
import torch
import torch_geometric
from torch_geometric.data import DataLoader
import glob
import os
import pandas
from pprint import pprint
import schema
from tqdm import tqdm
from docopt import docopt
import matplotlib.pyplot as plt
import seaborn as sns
import subprocess

from deep_logic.utils.layer import prune_logic_layers
from algos.models import AlgorithmProcessor
from algos.hyperparameters import get_hyperparameters
from algos.utils import iterate_over, plot_decision_trees, load_algorithms_and_datasets

def get_concept_labels(algoname):
    if algoname == 'BFS':
        return {'C1': 'hasBeenVisited', 'C2': 'hasVisitedNeighbour'}
    if algoname == 'parallel_coloring':
        return {
            'C1': 'isColored', 'C2': 'hasPriority',
            'C3': 'color1Seen', 'C4': 'color2Seen',
            'C5': 'color3Seen', 'C6': 'color4Seen',
            'C7': 'color5Seen',
        }
    return {}

sns.set()
args = docopt(__doc__)
schema = schema.Schema({'--algos': schema.And(list, [lambda n: n in ['BFS', 'parallel_coloring']]),
                        '--help': bool,
                        '--pruned': bool,
                        '--has-GRU': bool,
                        '--use-seeds': bool,
                        '--no-next-step-pool': bool,
                        '--pooling': schema.And(str, lambda s: s in ['attention', 'predinet', 'mean', 'max']),
                        '--num-seeds': schema.Use(int),
                        '--num-nodes': schema.Use(int),
                        '--model-path': schema.Or(None, schema.Use(str))})
args = schema.validate(args)
print(args)

if not os.path.exists('./algos/figures/'):
    os.makedirs('./algos/figures/')

_DEVICE = get_hyperparameters()['device']
_DIM_LATENT = get_hyperparameters()['dim_latent']
processor = AlgorithmProcessor(
    _DIM_LATENT,
    bias=get_hyperparameters()['bias'],
    use_gru=args['--has-GRU'],
).to(_DEVICE)
_gnrtrs = get_hyperparameters()['generators']
load_algorithms_and_datasets(args['--algos'],
                             processor, {
                                 'split': 'test',
                                 'generators': _gnrtrs,
                                 'num_nodes': args['--num-nodes'],
                             },
                             use_TF=False, # not used when testing
                             use_concepts=True,
                             use_concepts_sv=True, # In order to get validation losses/accuracies,not used otherwise
                             drop_last_concept=False,
                             use_decision_tree=False,
                             get_attention=False,
                             pooling=args['--pooling'],
                             next_step_pool=not args['--no-next-step-pool'],
                             bias=get_hyperparameters()['bias'])

if args['--pruned']:
    for name, algorithm in processor.algorithms.items():
        if algorithm.use_concepts:
            algorithm.concept_decoder = prune_logic_layers(
                algorithm.concept_decoder,
                0,
                0,
                device=_DEVICE)
print(processor)
processor.load_state_dict(torch.load(args['--model-path']))
processor.eval()
processor.load_split('test', num_nodes=args['--num-nodes'], datapoints=80)

algo0 = processor.algorithms[args['--algos'][0]]

pcms = []
for el in tqdm(algo0.dataset):
    toprocess = [el]
    toprocess = [el for el in DataLoader(
                toprocess,
                batch_size=get_hyperparameters()['batch_size'],
                shuffle=processor.training,
                drop_last=False)][0]
    algo0.zero_validation_stats()
    algo0.zero_formulas_aggregation()
    algo0.zero_steps()
    algo0.zero_tracking_losses_and_statistics()
    algo0.process(toprocess)
    pcms.append(algo0.get_validation_accuracies()['per_concept_mean_step_acc'].cpu())
pcms = torch.stack(pcms, dim=0).numpy()
fig, axs = plt.subplots(1, len(pcms[0]), sharey=True)
for cn in range(len(pcms[0])):
    axs[cn].hist(pcms[:, cn], bins=np.arange(0, 1.03, 0.02))
    axs[cn].set_xlabel(f"concept {cn+1}")
plt.yscale('log')
plt.show()
