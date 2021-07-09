"""
Usage:
    train.py [options] 

Options:
    -h --help              Show this screen.

    --epochs EP            Number of epochs to train. [default: 100]

    --model-name MN        Name of the model when saving. [default: kruskal]

    --no-use-concepts      Train without the concept bottleneck. [default: False]

    --L1-loss              Add L1 loss to the concept decoders part.
                           Currently ignored param. [default: False]

    --num-nodes NN         Number of nodes in the training graphs. [default: 20]

    --seed S               Random seed to set. [default: 47]

    --plot-DT              Plot decision tree for Kruskal [default: False]
"""
import os
import time
from docopt import docopt
import schema

import torch
from pprint import pprint
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from torch_geometric.data import DataLoader

from algos.mst.datasets import generate_dataset
from algos.mst.models.kruskal import KruskalNetworkPL

def get_DT_data(loader):

    from sklearn.tree import DecisionTreeClassifier
    from sklearn.tree import plot_tree
    import matplotlib.pyplot as plt
    cto_decision_tree = DecisionTreeClassifier()
    concepts = []
    targets = []
    for batch in loader:
        for timestep in range(len(batch.edge_concepts[0])):
            mask_all_fake_edges = (batch.edge_concepts[:, :, timestep] == -1).all(dim=-1)
            concepts.append(batch.edge_concepts[:, :, timestep][~mask_all_fake_edges])
            targets.append(batch.edge_targets[:, :, timestep][~mask_all_fake_edges])
    concepts = torch.cat(concepts)
    targets = torch.cat(targets)

    plt.figure(figsize=(10,10))
    cto_decision_tree.fit(concepts.cpu(), targets.cpu())
    plot_tree(cto_decision_tree, feature_names=['lighterEdgesVisited', 'nodesInSameSet', 'edgeInMST'], class_names=['unselected', 'selected'], impurity=False, max_depth=4, proportion=True, fontsize=18)
    plt.savefig(f'./algos/mst/figures/Kruskals_CTO_DT.png', bbox_inches='tight', pad_inches=0)
    exit(0)


def main():

    args = docopt(__doc__)
    sch = schema.Schema({'--plot-DT': bool,
                            '--help': bool,
                            '--L1-loss': bool,
                            '--num-nodes': schema.Use(int),
                            '--no-use-concepts': bool,
                            '--model-name': schema.Or(None, schema.Use(str)),
                            '--seed': schema.Use(int),
                            '--epochs': schema.Use(int)})

    args = sch.validate(args)

    seed = 42
    seed_everything(seed)

    # params
    n_nodes = args['--num-nodes']
    batch_size = 32
    train_size = 2000
    val_size = 200
    test_size = 200
    bits = 10
    embedding_size = 32
    dim_feedforward = 128
    nhead = 1
    n_concepts = 3
    concept_names = ['lighter_edges_visited', 'nodes_in_same_set', 'edge_in_mst']
    n_outputs = 1

    train_data, mne1, _ = generate_dataset(n_graphs=train_size, batch_size=batch_size, n_nodes=n_nodes, bits=bits)
    val_data, mne2, _ = generate_dataset(n_graphs=val_size, batch_size=batch_size, n_nodes=n_nodes, bits=bits)
    test_data, mne3, _ = generate_dataset(n_graphs=test_size, batch_size=batch_size, n_nodes=n_nodes, bits=bits)
    max_n_edges = max(mne1, mne2, mne3)
    batch = next(iter(train_data))

    processor_out_channels = 32
    encoder_in_features = batch.x.shape[2] + processor_out_channels
    encoder_out_features = 32

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False, follow_batch=['edge_nodes_index', 'edge_concepts'])

    if args['--plot-DT']:
        get_DT_data(train_loader)

    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, follow_batch=['edge_nodes_index', 'edge_concepts'])
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, follow_batch=['edge_nodes_index', 'edge_concepts'])

    base_dir = f'./algos/serialised_models/mst/explainer'
    os.makedirs(base_dir, exist_ok=True)

    seed_everything(args['--seed'])
    checkpoint_callback = ModelCheckpoint(dirpath=base_dir, monitor='val_loss',
                                          save_top_k=-1, filename=args['--model-name']+'-{epoch:04d}')
    checkpoint_callback_best = ModelCheckpoint(dirpath=base_dir, monitor='test_last_step_acc_explainer_epoch', mode='max',
                                          save_top_k=1, filename='best_'+args['--model-name'])
    trainer = Trainer(max_epochs=args['--epochs'], gpus=1, auto_lr_find=True, deterministic=True,
                      check_val_every_n_epoch=1, default_root_dir=base_dir,
                      weights_save_path=base_dir, callbacks=[checkpoint_callback, checkpoint_callback_best], profiler=None)
    model = KruskalNetworkPL(encoder_in_features, encoder_out_features, processor_out_channels,
                             bits, embedding_size, dim_feedforward, nhead, n_concepts, n_outputs, n_nodes, max_n_edges,
                             optimizer='adamw', lr=0.0010, device='cuda',
                             concept_names=concept_names, topk_explanations=10, no_use_concepts=args['--no-use-concepts'])

    start = time.time()
    trainer.fit(model, train_loader, val_loader)
    model.freeze()
    if not args['--no-use-concepts']:
        _ = trainer.test(model, test_dataloaders=train_loader)
        model.extract_formulas()
        print("Explanations")
        print(model.explanations)
    model_results = trainer.test(model, test_dataloaders=test_loader)
    print("Accuracy WITHOUT applying formulas")
    pprint(model_results)
    if not args['--no-use-concepts']:
        model.use_formulas = True
        model_results = trainer.test(model, test_dataloaders=test_loader)
        
        print("Accuracy WITH applying formulas")
        pprint(model_results)

    end = time.time() - start


if __name__ == '__main__':
    main()
    exit(0)
