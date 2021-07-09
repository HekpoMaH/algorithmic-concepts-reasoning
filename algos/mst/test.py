"""
Usage:
    test.py [options]

Options:
    -h --help              Show this screen.

    --model-path MP        Path of the model to load

    --no-use-concepts      Test without the concept bottleneck. [default: False]

    --use-seeds            Use seeds for STD. It will automatically modify name as
                           appropriate. [default: False]

    --num-nodes NN         Number of nodes in the graphs to test on [default: 20]

    --num-seeds NS         How many different seeds to plot. [default: 5]

    --all-num-nodes        Just do all 20, 50, 100 nodes' tests. [default: False]
"""
from pprint import pprint
import os
from docopt import docopt
import schema

import torch
import shutil
from pytorch_lightning import seed_everything, Trainer
from torch_geometric.data import DataLoader

from algos.mst.datasets import generate_dataset
from algos.mst.models.kruskal import KruskalNetworkPL

def add_formulas(model, trainer, train_loader, max_n_edges):
    model.n_nodes = 8
    model.max_n_edges = max_n_edges
    _ = trainer.test(model, test_dataloaders=train_loader)
    model.extract_formulas()

def get_data(n_graphs, batch_size, n_nodes, seed=47):
    seed_everything(seed)
    loader, max_n_edges_test, _ = generate_dataset(n_graphs=n_graphs, batch_size=batch_size, n_nodes=n_nodes, bits=10)
    test_loader = DataLoader(loader, batch_size=batch_size, shuffle=False, follow_batch=['edge_nodes_index', 'edge_concepts'])
    return test_loader, max_n_edges_test

def test_model(model, trainer, test_loader=None, max_n_edges_test=None, n_nodes=None, no_formulas=False):
    global args
    if n_nodes is None:
        n_nodes = args['--num-nodes']
    model.n_nodes = n_nodes
    batch_size = 64 if n_nodes < 50 else 8
    if test_loader is None:
        test_loader, max_n_edges_test = get_data(200, batch_size, n_nodes)
    model.max_n_edges = max_n_edges_test
    model_results = trainer.test(model, test_dataloaders=test_loader)
    print("MR")
    pprint(model_results)
    if no_formulas:
        return model_results[0], None
    model.use_formulas = True
    model_results_f = trainer.test(model, test_dataloaders=test_loader)
    print("MRE")
    pprint(model_results)
    print("E", model.explanations)
    return model_results[0], model_results_f[0]

def combine_seed(LD):
    DL = {k: torch.tensor([dic[k] for dic in LD]) for k in LD[0]}
    accs = {}
    for k, v in DL.items():
        if 'epoch' not in k:
            continue
        accs[k+'_mean'] = torch.mean(v, dim=0)
        accs[k+'_std'] = torch.std(v, dim=0)
    return accs

def test_num_nodes(num_nodes, trainer, formula_loader, form_max_n_edges):
    global args
    batch_size = 64
    if num_nodes > 20:
        batch_size = 16
    if num_nodes > 50:
        batch_size = 8
    # batch_size = 64 if num_nodes < 50 else 8
    dataloader, max_n_edges_test = get_data(200, batch_size, num_nodes)
    per_seed_accs, per_seed_accs_f = [], []
    formulas = {0: {}, 1: {}}
    for seed in range(args['--num-seeds']):
        model_path = args['--model-path'] + f'_seed_{seed}.ckpt'
        model = KruskalNetworkPL.load_from_checkpoint(model_path)
        model.freeze()
        if not args['--no-use-concepts']:
            add_formulas(model, trainer, formula_loader, form_max_n_edges)
            for i in range(2):
                if model.explanations[i] not in formulas[i]:
                    formulas[i][model.explanations[i]] = 0
                formulas[i][model.explanations[i]] += 1

        accs, accs_f = test_model(model, trainer, test_loader=dataloader, max_n_edges_test=max_n_edges_test, n_nodes=num_nodes, no_formulas=args['--no-use-concepts'])
        per_seed_accs.append(accs)
        per_seed_accs_f.append(accs_f)
    per_seed_accs = combine_seed(per_seed_accs)
    if not args['--no-use-concepts']:
        per_seed_accs_f = combine_seed(per_seed_accs_f)
    return per_seed_accs, per_seed_accs_f, formulas

def main():

    global args

    os.makedirs('./.temporaldirforPyL2', exist_ok=True)
    trainer = Trainer(max_epochs=4000, gpus=1, auto_lr_find=True, deterministic=True,
                      check_val_every_n_epoch=10, default_root_dir='./.temporaldirforPyL',
                      weights_save_path='./.temporaldirforPyL', callbacks=[], profiler=None)
    formula_loader, form_max_n_edges = get_data(2000, 64, 8, seed=42)
    if not args['--use-seeds']:
        model = KruskalNetworkPL.load_from_checkpoint(args['--model-path'])
        model.freeze()
        if not args['--no-use-concepts']:
            add_formulas(model, trainer, formula_loader, form_max_n_edges)
        test_model(model, trainer, no_formulas=args['--no-use-concepts'])
    elif not args['--all-num-nodes']:
        per_seed_accs, per_seed_accs_f, formulas = test_num_nodes(args['--num-nodes'], trainer, formula_loader, form_max_n_edges)
        print("WITHOUT FORMULAS")
        pprint(per_seed_accs)
        print("WITH FORMULAS")
        pprint(per_seed_accs_f)
        print('formulas')
        pprint(formulas)

    elif args['--all-num-nodes']:
        algo_per_seed_accs_per_num_nodes = {}
        algo_per_seed_accs_f_per_num_nodes = {}
        for nn in [20, 50, 100]:
            print("NN", nn)
            per_seed_accs, per_seed_accs_f, formulas = test_num_nodes(nn, trainer, formula_loader, form_max_n_edges)
            algo_per_seed_accs_per_num_nodes[nn] = per_seed_accs
            algo_per_seed_accs_f_per_num_nodes[nn] = per_seed_accs_f

        pprint(algo_per_seed_accs_per_num_nodes)
        pprint(algo_per_seed_accs_f_per_num_nodes)

        def print_metric(metric, dict_name, dic):
                    print(
                    f'& {metric} & '
                    f'${round(dic[20][f"{dict_name}"+"_mean"].item()*100, 2)}'
                    f'{{\scriptstyle\pm {round(dic[20][f"{dict_name}"+"_std"].item()*100, 2)}\%}}$ '
                    f'& ${round(dic[50][f"{dict_name}"+"_mean"].item()*100, 2)}'
                    f'{{\scriptstyle\pm {round(dic[50][f"{dict_name}"+"_std"].item()*100, 2)}\%}}$ '
                    f'& ${round(dic[100][f"{dict_name}"+"_mean"].item()*100, 2)}'
                    f'{{\scriptstyle\pm {round(dic[100][f"{dict_name}"+"_std"].item()*100, 2)}\%}}$ \\\\')

        print_metric('mean-step acc', 'test_mean_step_acc_explainer_epoch', algo_per_seed_accs_per_num_nodes)
        print_metric('last-step acc', 'test_last_step_acc_explainer_epoch', algo_per_seed_accs_per_num_nodes)
        if not args['--no-use-concepts']:
            print_metric('formula mean-step acc', 'test_mean_step_acc_explainer_epoch', algo_per_seed_accs_f_per_num_nodes)
            print_metric('formula last-step acc', 'test_last_step_acc_explainer_epoch', algo_per_seed_accs_f_per_num_nodes)
        print_metric('concepts mean-step acc', 'test_mean_step_acc_concepts_epoch', algo_per_seed_accs_per_num_nodes)
        print_metric('concepts last-step acc', 'test_last_step_acc_concepts_epoch', algo_per_seed_accs_per_num_nodes)

        pprint(formulas)

    shutil.rmtree('./.temporaldirforPyL2')


if __name__ == '__main__':

    global args
    args = docopt(__doc__)
    sch = schema.Schema({
                        '--help': bool,
                        '--use-seeds': bool,
                        '--all-num-nodes': bool,
                        '--no-use-concepts': bool,
                        '--model-path': schema.Use(str),
                        '--num-seeds': schema.Use(int),
                        '--num-nodes': schema.Use(int),
                        })

    args = sch.validate(args)

    main()
    exit(0)
