"""
Script to train models.

Usage:
    train.py [--algos=ALGO]... [options] 

Options:
    -h --help              Show this screen.

    --use-TF               Use Teacher Forcing or not during training. Not
                           using it, would add a GRU cell at the GNN update step.
                           [default: False]

    --use-GRU              Force the usage of a GRU cell at the GNN update step.
                           [default: False]

    --no-use-concepts      Do NOT utilise concepts bottleneck. If you set this
                           flag, set below flag too. [default: False]

    --no-use-concepts-sv   Do NOT utilise concepts supervision. [default: False]

    --pooling=PL           What graph pooling mechanism to use for termination.
                           One of {attention, predinet, max, mean}. [default: predinet]

    --no-next-step-pool    Do NOT use next step information for termination.
                           Use current instead. [default: False]

    --algos ALGO           Which algorithms to train {BFS, parallel_coloring}.
                           Repeatable parameter. [default: BFS]

    --epochs EP            Number of epochs to train. [default: 250]

    --no-patience          Do not utilise patience, train for max epochs. [default: False]

    --model-name MN        Name of the model when saving. Defaults to current time
                           and date if not provided.

    --L1-loss              Add L1 loss to the concept decoders part. [default: False]

    --prune-epoch PE       The epoch on which to prune logic layers.
                           The default of -1 does no pruning at all. [default: -1]

    --use-decision-tree    Use decision tree for concept->output mapping. [default: False]

    --drop-last-concept    Drop last concept? (Works only for coloring) [default: False]

    --seed S               Random seed to set. [default: 47]
"""

# TODO: consider adding a requirements.txt file

import json
import os
import torch
import torch.optim as optim
import random
import numpy as np
import deep_logic
import sympy
import schema
import copy
from deep_logic.utils.layer import prune_logic_layers
from datetime import datetime
from docopt import docopt
from algos.models import AlgorithmProcessor
from algos.hyperparameters import get_hyperparameters
from algos.utils import iterate_over, plot_decision_trees, load_algorithms_and_datasets
from pprint import pprint
from tqdm import tqdm

# torch.autograd.set_detect_anomaly(True)

args = docopt(__doc__)
schema = schema.Schema({'--algos': schema.And(list, [lambda n: n in ['BFS', 'parallel_coloring']]),
                        '--help': bool,
                        '--use-TF': bool,
                        '--L1-loss': bool,
                        '--use-decision-tree': bool,
                        '--no-use-concepts-sv': bool,
                        '--no-use-concepts': bool,
                        '--pooling': schema.And(str, lambda s: s in ['attention', 'predinet', 'mean', 'max']),
                        '--no-next-step-pool': bool,
                        '--use-GRU': bool,
                        '--no-patience': bool,
                        '--drop-last-concept': bool,
                        '--model-name': schema.Or(None, schema.Use(str)),
                        '--prune-epoch': schema.Use(int),
                        '--seed': schema.Use(int),
                        '--epochs': schema.Use(int)})

args = schema.validate(args)

NAME = args["--model-name"] if args["--model-name"] is not None else datetime.now().strftime("%b-%d-%Y-%H-%M")

torch.manual_seed(args['--seed'])
random.seed(args['--seed'])
np.random.seed(args['--seed'])
torch.cuda.manual_seed(0)
print("SEEDED with", args['--seed'])

_DEVICE = get_hyperparameters()['device']
_DIM_LATENT = get_hyperparameters()['dim_latent']

processor = AlgorithmProcessor(
    _DIM_LATENT,
    bias=get_hyperparameters()['bias'],
    prune_logic_epoch=args['--prune-epoch'],
    use_gru=not args['--use-TF'] or args['--use-GRU'],
).to(_DEVICE)

_gnrtrs = get_hyperparameters()['generators']
if 'parallel_coloring' in args['--algos']:
    _gnrtrs += ['deg5']
print("GENERATORS", _gnrtrs)
load_algorithms_and_datasets(args['--algos'],
                             processor, {
                                 'split': 'train',
                                 'generators': _gnrtrs,
                                 'num_nodes': 20,
                             },
                             use_TF=args['--use-TF'],
                             use_concepts=not args['--no-use-concepts'],
                             use_concepts_sv=not args['--no-use-concepts-sv'],
                             drop_last_concept=args['--drop-last-concept'],
                             pooling=args['--pooling'],
                             next_step_pool=not args['--no-next-step-pool'],
                             L1_loss=args['--L1-loss'],
                             prune_logic_epoch=args['--prune-epoch'],
                             use_decision_tree=args['--use-decision-tree'],
                             new_coloring_dataset=False,
                             bias=get_hyperparameters()['bias'])



best_model = copy.deepcopy(processor)
best_score = float('inf')

print(processor)
term_params = []
normal_params = []
for name, param in processor.named_parameters():
    if '_term' in name or 'termination' in name or 'predinet' in name:
        term_params.append(param)
    else:
        normal_params.append(param)
lr = get_hyperparameters()[f'lr']
optimizer = optim.Adam([
                           {'params': term_params, 'lr': lr},
                           {'params': normal_params, 'lr': lr}
                       ],
                       lr=get_hyperparameters()[f'lr'],
                       weight_decay=get_hyperparameters()['weight_decay'])

patience = 0
hardcode_outputs = False
hardcoding = torch.zeros(list(processor.algorithms.values())[0].concept_features, dtype=torch.bool).to(_DEVICE)
hardcoding[0] = False
hardcoding[1] = False
hardcoding = None

for epoch in range(args['--epochs']):
    processor.load_split('train')
    processor.train()
    iterate_over(processor, optimizer=optimizer, epoch=epoch, hardcode_concepts=hardcoding, hardcode_outputs=hardcode_outputs)
    if epoch == processor.prune_logic_epoch:
        best_score = float('inf')
        for name, algorithm in processor.algorithms.items():
            if algorithm.use_concepts:
                algorithm.concept_decoder = prune_logic_layers(
                    algorithm.concept_decoder,
                    epoch,
                    algorithm.prune_logic_epoch,
                    device=_DEVICE)

    serialised_models_dir = './algos/serialised_models/'
    if not os.path.isdir(serialised_models_dir):
        os.makedirs(serialised_models_dir)
    if (epoch + 1) % 10 == 0:
        torch.save(processor.state_dict(), './algos/serialised_models/test_'+NAME+'_epoch_'+str(epoch)+'.pt')
    processor.eval()
    if (epoch+1) % 1 == 0:
        processor.eval()
        print("EPOCH", epoch)
        for spl in ['val']:
            print("SPLIT", spl)
            processor.load_split(spl)
            iterate_over(processor, epoch=epoch, hardcode_concepts=hardcoding, hardcode_outputs=hardcode_outputs)
            total_sum = 0
            for name, algorithm in processor.algorithms.items():
                print("algo", name)
                pprint(algorithm.get_losses_dict(validation=True))
                pprint(algorithm.get_validation_accuracies())
                total_sum += sum(algorithm.get_losses_dict(validation=True).values())
            if spl == 'val':
                patience += 1
            if total_sum < best_score and spl == 'val':
                best_score = total_sum
                best_model = copy.deepcopy(processor)
                torch.save(best_model.state_dict(), './algos/serialised_models/best_'+NAME+'.pt')
                patience = 0

            total_sum2 = 0

        print("PATIENCE", patience, total_sum, total_sum2, best_score)
    if patience >= 50 and not args['--no-patience']:
        break

torch.save(best_model.state_dict(), './algos/serialised_models/best_'+NAME+'.pt')

if args['--use-decision-tree']:
    iterate_over(best_model, fit_decision_tree=True, hardcode_concepts=hardcoding)
    plot_decision_trees(best_model.algorithms)

print("TESTING!!!")
if not args['--no-use-concepts'] and not args['--use-decision-tree']:
    iterate_over(best_model, extract_formulas=True, epoch=0, hardcode_concepts=hardcoding, hardcode_outputs=hardcode_outputs)
    for algorithm in best_model.algorithms.values():
        print("EXPLANATIONS", algorithm.explanations)
best_model.eval()
best_model.load_split('test')
iterate_over(best_model, apply_formulas=True and not args['--no-use-concepts'] and not args['--use-decision-tree'], apply_decision_tree=args['--use-decision-tree'], epoch=0, hardcode_concepts=hardcoding, hardcode_outputs=hardcode_outputs)
for algorithm in best_model.algorithms.values():
    pprint(algorithm.get_validation_accuracies())

