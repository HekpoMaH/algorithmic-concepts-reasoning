"""
Usage:
    test_several_seeds.py [options]

Options:
    -h --help              Show this screen.

    --model-name MN        Name of the model when saving.

    --num-nodes NN         Number of nodes in the graphs to test on [default: 20]

    --test-formulas        Whether to test formulas on each epoch. If not set,
                           formula results will be set to -1 in the respective columns.
                           BEWARE: This slows down the process significantly.
                           [default: False]
  
    --num-seeds NS         How many different seeds to test. [default: 5]

    --CUDA-START CS        From which cuda device to start [default: 0]

    --CUDA-MOD CM          How many machines to cycle [default: 1]

"""
import os
import schema
from docopt import docopt

args = docopt(__doc__)
schema = schema.Schema({
                        '--help': bool,
                        '--num-seeds': schema.Use(int),
                        '--num-nodes': schema.Use(int),
                        '--test-formulas': bool,
                        '--CUDA-MOD': schema.Use(int),
                        '--CUDA-START': schema.Use(int),
                        '--model-name': schema.Or(None, schema.Use(str))})
args = schema.validate(args)

commands = []
for seed in range(args['--num-seeds']):
    machine = args['--CUDA-START'] + seed % args['--CUDA-MOD']
    command = f'CUDA_VISIBLE_DEVICES={machine} python -m algos.mst.test_per_epoch --model-name {args["--model-name"]}_seed_{seed} '

    for flag in [
            '--test-formulas',
    ]:
        if args[flag]:
            command += flag + ' '
    command += f'--num-nodes {args["--num-nodes"]} '
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
