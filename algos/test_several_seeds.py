"""
Script to test a model **on every epoch** that is serialised in the serialised
models' folder.

Usage:
    test_several_seeds.py [--algos=ALGO]... [options]

Options:
    -h --help              Show this screen.

    --has-GRU              Does the processor utilise GRU unit? [default: False]

    --pooling=PL           What graph pooling mechanism to use for termination.
                           One of {attention, max, mean, predinet}. [default: predinet]

    --no-next-step-pool    Do NOT use next step information for termination.
                           Use current instead. [default: False]

    --algos ALGO           Which algorithms to train {BFS, parallel_coloring}.
                           Repeatable parameter. [default: BFS]

    --model-name MN        Name of the model when saving.

    --num-nodes NN         Number of nodes in the graphs to test on [default: 20]

    --test-formulas        Whether to test formulas on each epoch. If not set,
                           formula results will be set to -1 in the respective columns.
                           BEWARE: This slows down the process significantly.
                           [default: False]
  
    --num-seeds NS         How many different seeds to test. [default: 5]

    --prune-epoch PE       At which epoch was pruning applied. The default value
                           results in no pruning epoch. [default: 1000000000]

    --no-use-concepts      Do NOT utilise concepts bottleneck. [default: False]

    --drop-last-concept    Drop last concept? (Works only for coloring) [default: False]

    --CUDA-START CS        From which cuda device to start [default: 0]

    --CUDA-MOD CM          How many machines to cycle [default: 1]

"""
import os
import schema
from docopt import docopt

args = docopt(__doc__)
schema = schema.Schema({'--algos': schema.And(list, [lambda n: n in ['BFS', 'parallel_coloring']]),
                        '--help': bool,
                        '--has-GRU': bool,
                        '--num-seeds': schema.Use(int),
                        '--num-nodes': schema.Use(int),
                        '--no-use-concepts': bool,
                        '--drop-last-concept': bool,
                        '--test-formulas': bool,
                        '--pooling': schema.And(str, lambda s: s in ['attention', 'mean', 'max', 'predinet']),
                        '--no-next-step-pool': bool,
                        '--prune-epoch': schema.Use(int),
                        '--CUDA-MOD': schema.Use(int),
                        '--CUDA-START': schema.Use(int),
                        '--model-name': schema.Or(None, schema.Use(str))})
args = schema.validate(args)

commands = []
for seed in range(args['--num-seeds']):
    machine = args['--CUDA-START'] + seed % args['--CUDA-MOD']
    command = f'CUDA_VISIBLE_DEVICES={machine} python -m algos.test_per_epoch --model-name {args["--model-name"]}_seed_{seed} --pooling {args["--pooling"]} '

    for algo in args['--algos']:
        command += f'--algos {algo} '

    for flag in [
            '--has-GRU', '--test-formulas', '--no-use-concepts', '--drop-last-concept',
            '--no-next-step-pool'
    ]:
        if args[flag]:
            command += flag + ' '
    command += f'--num-nodes {args["--num-nodes"]} --prune-epoch {args["--prune-epoch"]} '
    command += '&'
    commands.append(command)
print(commands)

with open("runseeds.sh", 'w+') as f:
    for command in commands:
        print(command, file=f)
    pass

os.system('chmod +x runseeds.sh')
os.system('./runseeds.sh')
os.system('rm runseeds.sh')
