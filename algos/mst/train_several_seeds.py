"""
Usage:
    train_several_seeds.py [options]

Options:
    -h --help              Show this screen.

    --epochs EP            Number of epochs to train. [default: 100]

    --no-use-concepts      Train without the concept bottleneck. [default: False]

    --model-name MN        Name of the model when saving. [default: kruskal]

    --L1-loss              Add L1 loss to the concept decoders part. 
                           Currently ignored param. [default: False]

    --num-seeds NS         How many different seeds to train [default: 5]

    --num-nodes NN         Number of nodes in the training graphs. [default: 20]

    --starting-seed SS     Which seed index to start from [default: 0]

    --CUDA-START CS        From which cuda device to start [default: 0]

    --CUDA-MOD CM          How many machines to cycle [default: 1]
"""
import os
from docopt import docopt
import schema

args = docopt(__doc__)
schema = schema.Schema({
                        '--help': bool,
                        '--no-use-concepts': bool,
                        '--L1-loss': bool,
                        '--num-nodes': schema.Use(int),
                        '--num-seeds': schema.Use(int),
                        '--starting-seed': schema.Use(int),
                        '--CUDA-MOD': schema.Use(int),
                        '--CUDA-START': schema.Use(int),
                        '--model-name': schema.Or(None, schema.Use(str)),
                        '--epochs': schema.Use(int)})
args = schema.validate(args)
print(args)

commands = []
for seed in range(args['--starting-seed'], args['--starting-seed']+args['--num-seeds']):
    machine = args['--CUDA-START'] + seed % args['--CUDA-MOD']
    command = f'CUDA_VISIBLE_DEVICES={machine} python -m algos.mst.train --epochs {args["--epochs"]} --num-nodes {args["--num-nodes"]} --model-name {args["--model-name"]}_seed_{seed} '

    for flag in [
            '--L1-loss', '--no-use-concepts',
    ]:
        if args[flag]:
            command += flag + ' '
    command += f'--seed {seed} '
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
