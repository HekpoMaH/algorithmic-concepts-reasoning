"""

Script to test a model on every epoch that is serialised in the serialised
models' folder. Decision which model-epoch combination to test is performed
based on provided model name.

Usage:
    test_per_epoch.py (--model-name MN) [--algos=ALGO]... [options] 

Options:
    -h --help              Show this screen.

    --algos ALGO           Which algorithms to load {BFS, parallel_coloring}.
                           Repeatable parameter. [default: BFS]

    --has-GRU              Does the processor utilise GRU unit? [default: False]

    --pooling=PL           What graph pooling mechanism to use for termination.
                           One of {attention, max, mean, predinet}. [default: predinet]

    --no-next-step-pool    Do NOT use next step information for termination.
                           Use current instead. [default: False]

    --model-name MN        Name of the model to load

    --num-nodes NN         Number of nodes in the graphs to test on [default: 20]

    --test-formulas        Whether to test formulas on each epoch. If not set,
                           formula results will be set to -1 in the respective columns.
                           BEWARE: This slows down the process significantly.
                           [default: False]

    --no-use-concepts      Do NOT utilise concepts bottleneck. [default: False]

    --prune-epoch PE       At which epoch was pruning applied. The default value
                           results in no pruning epoch. [default: 1000000000]
"""
import glob
import os
from pprint import pprint
import torch
import schema
from tqdm import tqdm
from docopt import docopt
from deep_logic.utils.layer import prune_logic_layers
from algos.models import AlgorithmProcessor
from algos.hyperparameters import get_hyperparameters
from algos.utils import iterate_over, load_algorithms_and_datasets

def get_epoch(model_path):
    return int(model_path.split('epoch_')[1].split('.')[0])

args = docopt(__doc__)
schema = schema.Schema({'--algos': schema.And(list, [lambda n: n in ['BFS', 'parallel_coloring']]),
                        '--help': bool,
                        '--has-GRU': bool,
                        '--test-formulas': bool,
                        '--no-use-concepts': bool,
                        '--pooling': schema.And(str, lambda s: s in ['attention', 'mean', 'max', 'predinet']),
                        '--no-next-step-pool': bool,
                        '--num-nodes': schema.Use(int),
                        '--prune-epoch': schema.Use(int),
                        '--model-name': schema.Or(None, schema.Use(str))})
args = schema.validate(args)
print(args)

_DEVICE = get_hyperparameters()['device']
_DIM_LATENT = get_hyperparameters()['dim_latent']
_BATCH_SIZE = get_hyperparameters()['batch_size']
processor = AlgorithmProcessor(
    _DIM_LATENT,
    bias=get_hyperparameters()['bias'],
    use_gru=args['--has-GRU'],
    prune_logic_epoch=args['--prune-epoch'],
).to(_DEVICE)

_gnrtrs = get_hyperparameters()['generators']
if 'parallel_coloring' in args['--algos']:
    _gnrtrs += ['deg5']
load_algorithms_and_datasets(args['--algos'],
                             processor, {
                                 'split': 'test',
                                 'generators': _gnrtrs,
                                 'num_nodes': args['--num-nodes'],
                             },
                             use_concepts=not args['--no-use-concepts'],
                             use_concepts_sv=not args['--no-use-concepts'],
                             get_attention=True and args['--pooling'],
                             pooling=args['--pooling'],
                             next_step_pool=not args['--no-next-step-pool'],
                             prune_logic_epoch=args['--prune-epoch'],
                             bias=get_hyperparameters()['bias'])

patt = './algos/serialised_models/test_' + args['--model-name']+'*'
if not os.path.exists('./algos/results/'):
    os.makedirs('./algos/results/')

model_paths = glob.glob(patt)
model_paths = sorted(model_paths, key=get_epoch)
assert model_paths, 'Check model name, no such models found'
print(model_paths)

accs_per_epoch = {}
losses_per_epoch = {}
savepath = './algos/results/'+args['--model-name']+'.csv'
print(savepath)
algos = list(processor.algorithms.keys())
epochs = []
print(algos)
processor.eval()
pruned = False
for i, mp in enumerate(tqdm(model_paths)):
    epoch = int(get_epoch(mp))
    epochs.append(epoch)
    if epoch >= args['--prune-epoch'] and not pruned:
        pruned = True
        for name, algorithm in processor.algorithms.items():
            if algorithm.use_concepts:
                algorithm.concept_decoder = prune_logic_layers(
                    algorithm.concept_decoder,
                    algorithm.prune_logic_epoch,
                    algorithm.prune_logic_epoch,
                    device=_DEVICE)

    processor.load_state_dict(torch.load(mp))
    if args['--test-formulas'] and not args['--no-use-concepts']:
        processor.load_split('train')
        iterate_over(processor, epoch=0, extract_formulas=True, batch_size=_BATCH_SIZE)
    processor.load_split('test')
    iterate_over(processor, epoch=0, apply_formulas=False, batch_size=_BATCH_SIZE)
    accs_per_epoch[epoch] = {}
    losses_per_epoch[epoch] = {}
    for algorithm in algos:
        curr_algo = processor.algorithms[algorithm]
        accs_per_epoch[epoch][algorithm] = curr_algo.get_validation_accuracies()
        losses_per_epoch[epoch][algorithm] = curr_algo.get_losses_dict(validation=True)
    if args['--test-formulas'] and not args['--no-use-concepts']:
        iterate_over(processor, epoch=0, apply_formulas=True, batch_size=_BATCH_SIZE)
    for algorithm in algos:
        curr_algo = processor.algorithms[algorithm]
        val_acc = curr_algo.get_validation_accuracies()
        for acc in [
                'formula_last_step_acc', 'formula_mean_step_acc',
                'term_formula_mean_step_acc', 'per_concept_mean_step_acc',
                'per_concept_last_step_acc'
        ]:
            if 'formula' in acc and not args['--test-formulas']:
                accs_per_epoch[epoch][algorithm][acc] = torch.tensor(-1.)
            else:
                accs_per_epoch[epoch][algorithm][acc] = val_acc[acc]

pprint(accs_per_epoch)
pprint(losses_per_epoch)

topline = 'epoch,'
for algo in algos:
    topline += f'{algo}_loss,'
    topline += f'{algo}_mean_step_acc,'
    topline += f'{algo}_last_step_acc,'
    topline += f'{algo}_concepts_mean_step_acc,'
    topline += f'{algo}_concepts_last_step_acc,'
    topline += f'{algo}_formula_mean_step_acc,'
    topline += f'{algo}_formula_last_step_acc,'
    topline += f'{algo}_term_formula_mean_step_acc,'
    for i in range(processor.algorithms[algo].concept_features):
        topline += f'{algo}_concept_{i}_mean_step_acc,'
    for i in range(processor.algorithms[algo].concept_features):
        topline += f'{algo}_concept_{i}_last_step_acc,'
    topline = topline[:-1]

with open(savepath, "w") as f:
    print(topline, file=f)
    for epoch in sorted(epochs):
        print(str(epoch)+',', end='', file=f)
        for algo in algos:
            print(str(sum(losses_per_epoch[epoch][algo].values()).item())+',', end='', file=f)
            for acc in [
                    'mean_step_acc', 'last_step_acc', 'concepts_mean_step_acc',
                    'concepts_last_step_acc', 'formula_mean_step_acc',
                    'formula_last_step_acc', 'term_formula_mean_step_acc',
                    'per_concept_mean_step_acc', 'per_concept_last_step_acc'
            ]:
                app = ',' if acc != 'per_concept_last_step_acc' else ''
                if not torch.is_tensor(accs_per_epoch[epoch][algo][acc]):
                    accs_per_epoch[epoch][algo][acc] = torch.tensor(accs_per_epoch[epoch][algo][acc])
                print(str(accs_per_epoch[epoch][algo][acc].view(-1).tolist())[1:-1] + app, end='', file=f)
        print(file=f)
