"""
Usage:
    test_per_epoch.py (--model-name MN) [options] 

Options:
    -h --help           Show this screen.

    --model-name MN     Name of the model to load

    --num-nodes NN      Number of nodes in the graphs to test on [default: 20]

    --test-formulas     Whether to test formulas on each epoch. If not set,
                        formula results will be set to -1 in the respective columns.
                        BEWARE: This slows down the process significantly, so use
                        when you have very few epochs to test.
                        [default: False]
"""
import shutil
import glob
import os
import torch
import schema
from docopt import docopt

from pytorch_lightning import Trainer

from algos.mst.models.kruskal import KruskalNetworkPL
from algos.mst.test import get_data, test_model
def get_epoch(model_path):
    return int(model_path.split('epoch=')[1].split('.')[0])

def main():
    args = docopt(__doc__)
    sch = schema.Schema({
                            '--help': bool,
                            '--test-formulas': bool,
                            '--num-nodes': schema.Use(int),
                            '--model-name': schema.Or(None, schema.Use(str))})
    args = sch.validate(args)
    print(args)
    basedir = f'./.temporaldirforPyL{args["--model-name"]}'
    os.makedirs(basedir, exist_ok=True)

    patt = f'./algos/serialised_models/mst/explainer/{args["--model-name"]}-epoch=????.ckpt'
    if not os.path.exists('./algos/results/'):
        os.makedirs('./algos/results/')
    model_paths = glob.glob(patt)
    model_paths = sorted(model_paths, key=get_epoch)

    trainer = Trainer(max_epochs=4000, gpus=1, auto_lr_find=True, deterministic=True,
                      check_val_every_n_epoch=10, default_root_dir=basedir,
                      weights_save_path=basedir, callbacks=[], profiler=None)
    test_loader, test_max_n_edges = get_data(200, 64, args['--num-nodes'], seed=47)

    accs_per_epoch = {}
    losses_per_epoch = {}
    epochs = []
    print("FOUND", model_paths)
    for model_path in model_paths:
        epoch = get_epoch(model_path)
        epochs.append(epoch)
        model = KruskalNetworkPL.load_from_checkpoint(model_path)
        model.freeze()

        accs = test_model(model,
                          trainer,
                          test_loader=test_loader,
                          max_n_edges_test=test_max_n_edges,
                          n_nodes=args['--num-nodes'],
                          no_formulas=True)
        accs_per_epoch[epoch], _ = accs

    savepath = './algos/results/'+args['--model-name']+'.csv'
    topline = 'epoch,'
    for algo in ['kruskal']:
        topline += f'{algo}_loss,'
        topline += f'{algo}_mean_step_acc,'
        topline += f'{algo}_last_step_acc,'
        topline += f'{algo}_concepts_mean_step_acc,'
        topline += f'{algo}_concepts_last_step_acc,'
        topline += f'{algo}_formula_mean_step_acc,'
        topline += f'{algo}_formula_last_step_acc,'
        topline += f'{algo}_term_formula_mean_step_acc,'
        for i in range(3):
            topline += f'{algo}_concept_{i}_mean_step_acc,'
        for i in range(3):
            topline += f'{algo}_concept_{i}_last_step_acc,'
        topline = topline[:-1]

    with open(savepath, "w") as f:
        print(topline, file=f)
        for epoch in sorted(epochs):
            print(str(epoch+1)+',', end='', file=f)
            print(str(0)+',', end='', file=f)
            for acc in [
                    'test_mean_step_acc_explainer_epoch', 'test_last_step_acc_explainer_epoch', 'test_mean_step_acc_concepts_epoch',
                    'test_last_step_acc_concepts_epoch',
            ]:
                app = ','
                if not torch.is_tensor(accs_per_epoch[epoch][acc]):
                    accs_per_epoch[epoch][acc] = torch.tensor(accs_per_epoch[epoch][acc])
                print(str(accs_per_epoch[epoch][acc].view(-1).tolist())[1:-1] + app, end='', file=f)
            print("-1.0,-1.0,-1.0,", end='', file=f)
            apee = accs_per_epoch[epoch]
            print(f"{apee['test_mean_step_acc_concepts_0_epoch']},", end='', file=f)
            print(f"{apee['test_mean_step_acc_concepts_1_epoch']},", end='', file=f)
            print(f"{apee['test_mean_step_acc_concepts_2_epoch']},", end='', file=f)
            print(f"{apee['test_last_step_acc_concepts_0_epoch']},", end='', file=f)
            print(f"{apee['test_last_step_acc_concepts_1_epoch']},", end='', file=f)
            print(f"{apee['test_last_step_acc_concepts_2_epoch']},", end='', file=f)
            print(file=f)

    shutil.rmtree(basedir)

if __name__ == '__main__':
    main()
