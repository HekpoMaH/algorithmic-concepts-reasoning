import itertools
from pprint import pprint
import numpy as np

import torch
import torch.optim as optim
import torch_scatter
from torch_geometric.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree

import deep_logic

from algos.datasets import BFSSingleIterationDataset, ParallelColoringSingleGeneratorDataset, ParallelColoringDataset, CombinedGeneratorsDataset, TerminationCombinationDataset
from algos.hyperparameters import get_hyperparameters
import algos.models as models

def flip_edge_index(edge_index):
    return torch.stack((edge_index[1], edge_index[0]), dim=0)

_DEVICE = get_hyperparameters()['device']
_DIM_LATENT = get_hyperparameters()['dim_latent']

def prepare_batch(batch):
    batch = batch.to(_DEVICE)
    if len(batch.x.shape) == 2:
        batch.x = batch.x.unsqueeze(-1)
    batch.x = batch.x.transpose(1, 0)
    batch.y = batch.y.transpose(1, 0)
    batch.termination = batch.termination.transpose(1, 0)
    batch.concepts = batch.concepts.transpose(1, 0)
    batch.concepts_fin = batch.concepts_fin.transpose(1, 0)
    batch.num_nodes = len(batch.x[0])
    return batch

def get_graph_embedding(latent_nodes, batch_ids, reduce='mean'):
    graph_embs = torch_scatter.scatter(latent_nodes, batch_ids, dim=0, reduce=reduce)
    return graph_embs

def _test_get_graph_embedding():
    latent_nodes = torch.tensor([[1., 5.], [3., 8.], [15., 20.]])
    batch_ids = torch.tensor([0, 0, 1])
    assert torch.allclose(get_graph_embedding(latent_nodes, batch_ids), torch.tensor([[2., 6.5], [15., 20.]]))

def get_mask_to_process(continue_logits, batch_ids, debug=False):
    """

    Used for graphs with different number of steps needed to be performed

    Returns:
    mask (1d tensor): The mask for which nodes still need to be processed

    """
    if debug:
        print("Getting mask processing")
        print("Continue logits:", continue_logits)
    mask = continue_logits[batch_ids] > 0
    if debug:
        print("Mask:", mask)
    return mask

def _test_get_mask_to_process():
    cp = torch.tensor([0.78, -0.22])
    batch_ids = torch.tensor([0, 0, 1])
    assert (get_mask_to_process(cp, batch_ids, debug=True) == torch.tensor([True, True, False])).all()


def load_algorithms_and_datasets(algorithms,
                                 processor,
                                 dataset_kwargs,
                                 bias=False,
                                 use_TF=False,
                                 use_concepts=True,
                                 use_concepts_sv=True,
                                 drop_last_concept=False,
                                 L1_loss=False,
                                 prune_logic_epoch=-1,
                                 use_decision_tree=False,
                                 pooling='attention',
                                 next_step_pool=True,
                                 new_coloring_dataset=False,
                                 get_attention=False):
    for algorithm in algorithms:
        algo_class = models.AlgorithmBase if algorithm == 'BFS' else models.AlgorithmColoring
        inside_class = BFSSingleIterationDataset if algorithm == 'BFS' else ParallelColoringSingleGeneratorDataset
        dataclass = CombinedGeneratorsDataset if algorithm == 'BFS' or new_coloring_dataset else ParallelColoringDataset
        rootdir = f'./algos/{algorithm}'
        algo_net = algo_class(get_hyperparameters()[f'dim_latent'],
                              get_hyperparameters()[f'dim_nodes_{algorithm}'],
                              get_hyperparameters()[f'dim_concept_{algorithm}'],
                              get_hyperparameters()[f'dim_target_{algorithm}'],
                              processor,
                              dataclass,
                              inside_class,
                              rootdir,
                              dataset_kwargs,
                              sigmoid=False,
                              bias=bias,
                              use_TF=use_TF,
                              use_concepts=use_concepts,
                              use_concepts_sv=use_concepts_sv,
                              drop_last_concept=drop_last_concept,
                              L1_loss=L1_loss,
                              prune_logic_epoch=prune_logic_epoch,
                              use_decision_tree=use_decision_tree,
                              global_termination_pool=pooling,
                              next_step_pool=next_step_pool,
                              get_attention=get_attention).to(_DEVICE)
        processor.add_algorithm(algo_net, algorithm)

def remove_empty_explanations(explanations):
    return {k: v if v else '(False)' for (k, v) in explanations.items()}

def fit_decision_trees_to_algorithms(algorithms):
    '''
    Given a list of algorithms, fit a decision tree for its
    concept -> output or concept -> termination mapping
    '''
    for name, algorithm in algorithms.items():
        algorithm.cto_decision_tree = DecisionTreeClassifier()

        # concatenate concepts/predictions from each batch into one tensor
        all_concepts = torch.cat(algorithm.actual['concepts'], dim=0)
        
        all_target = torch.cat(algorithm.actual['outputs'], dim=0)
        all_termination_target = torch.cat(algorithm.actual['terminations'], dim=0)

        algorithm.cto_decision_tree.fit(all_concepts.cpu(), all_target.cpu())

def plot_decision_trees(algorithms):
    f_names = {
        'parallel_coloring': ['isColored', 'hasPriority', 'color1Seen', 'color2Seen', 'color3Seen', 'color4Seen', 'color5Seen'],
        'BFS': ['hasBeenVisited', 'hasVisitedNeighbours']
    }
    c_names = {
        'parallel_coloring': ['uncolored', 'colored1', 'colored2', 'colored3', 'colored4', 'colored5'],
        'BFS': ['unvisited', 'visited']
    }
    for name, algorithm in algorithms.items():
        if name =='BFS':
            plt.figure(figsize=(10,10))
        else:
            plt.figure(figsize=(50,20))
        plot_tree(algorithm.cto_decision_tree, feature_names=f_names[name], class_names=c_names[name], impurity=False, max_depth=6, proportion=False, fontsize=18, label='none')
        plt.savefig(f'./algos/figures/{name}_CTO_DT.png', bbox_inches='tight', pad_inches=0)
        c_names = ['stop', 'continue']

def add_explanations_to_algorithms(algorithms):
    '''
    Given a list of algorithms, it explains each of the algorithm
    outputs and adds explanations to the corresponding algorithm
    '''

    def check_truth_assignment(truth_assignment, indexes, dataset):
        for combinations, termination in dataset:
            if not (truth_assignment == combinations[:, indexes]).all(axis=-1).any() == termination[0]:
                return False
        return True
    def get_pretty_explanation(indexes, truth_assignment):
        re = []

        for idx in indexes:
            str_i = f'{"" if truth_assignment[idx] else "~"}feature{0:09d}{idx}'
            re.append(str_i)
        return ' & '.join(re)

    for name, algorithm in algorithms.items():
        explanations = {i: {} for i in range(algorithm.output_features)}
        explanations.update({'termination': {}})

        # concatenate concepts/predictions from each batch into one tensor
        all_concepts = torch.cat(algorithm.predictions['concepts'], dim=0)
        all_termination_target = torch.cat(algorithm.actual['terminations'], dim=0)
        all_logits = torch.cat(algorithm.predictions['outputs'], dim=0)
        all_target = torch.cat(algorithm.actual['outputs'], dim=0)
        all_target_from_logits = (all_logits > 0).long if name == 'BFS' else all_logits.argmax(dim=-1)

        # After they are all stacked up
        # explain each class in turn
        for cls in range(max(algorithm.output_features, 2)):
            expl, expls = deep_logic.logic.explain_class(
                algorithm.concept_decoder,
                all_concepts,
                all_target,
                False if name=='parallel_coloring' else True,
                cls,
                simplify=True,
                topk_explanations=5000)
            explanations[cls] = expl

        # and explain termination
        algorithm.explanations = explanations

        # Add rules for termination
        t = TerminationCombinationDataset(algorithm.dataset)
        num_concepts = t.dataset[0][0].shape[1]
        indexes = [i for i in range(num_concepts)]
        explanation = None
        for l in range(1, num_concepts+1):
            for idx_combintion in itertools.combinations(indexes, l):
                for truth_assignment in itertools.product([0, 1], repeat=l):
                    if check_truth_assignment(np.array(truth_assignment), idx_combintion, t.dataset):
                        print("FOUND IT", torch.tensor(idx_combintion), torch.tensor(truth_assignment))
                        explanation = (np.array(truth_assignment), idx_combintion)
                        algorithm.termination_idx_combination = torch.tensor(idx_combintion).to(_DEVICE)
                        algorithm.termination_truth_assignment = torch.tensor(truth_assignment).to(_DEVICE)
                        break
                if explanation is not None:
                    break
            if explanation is not None:
                break

        
        algorithm.explanations['termination'] = get_pretty_explanation(algorithm.termination_idx_combination, algorithm.termination_truth_assignment)


def iterate_over(processor,
                 optimizer=None,
                 return_outputs=False,
                 num_classes=2,
                 num_concepts=2,
                 extract_formulas=False,
                 apply_formulas=False,
                 hardcode_concepts=None,
                 hardcode_outputs=False,
                 sigmoid=False,
                 epoch=None,
                 batch_size=None,
                 fit_decision_tree=False,
                 apply_decision_tree=False,
                 aggregate=False):

    done = {}
    iterators = {}
    for name, algorithm in processor.algorithms.items():
        algorithm.epoch = epoch
        iterators[name] = iter(DataLoader(
            algorithm.dataset,
            batch_size=get_hyperparameters()['batch_size'] if batch_size is None else batch_size,
            shuffle=False,
            drop_last=False))
        done[name] = False
        algorithm.zero_validation_stats()
        if extract_formulas or fit_decision_tree:
            algorithm.zero_formulas_aggregation()

    idx = 0
    while True:
        for name, algorithm in processor.algorithms.items():
            try:
                algorithm.zero_steps()
                algorithm.zero_tracking_losses_and_statistics()
                batch = next(iterators[name])
                with torch.set_grad_enabled(processor.training):
                    algorithm.process(batch,
                                      extract_formulas=extract_formulas,
                                      fit_decision_tree=fit_decision_tree,
                                      apply_formulas=apply_formulas,
                                      apply_decision_tree=apply_decision_tree,
                                      hardcode_concepts=hardcode_concepts,
                                      hardcode_outputs=hardcode_outputs)
                    # Wrong flag and wrong indices are collected
                    # if we want to check which batch/sample was misclassified
                    if algorithm.wrong_flag and not algorithm.training:
                        algorithm.wrong_indices.append(idx)

            # All algorithms are iterated for at most |nodes| steps
            except StopIteration:
                done[name] = True
                continue
        if processor.training and not all(done.values()):
            processor.update_weights(optimizer)
        idx += 1
        if all(done.values()):
            break

    if extract_formulas:
        add_explanations_to_algorithms(processor.algorithms)

    if fit_decision_tree:
        fit_decision_trees_to_algorithms(processor.algorithms)

def toggle_freeze_module(module):
    for param in module.parameters():
        param.requires_grad ^= True

if __name__ == '__main__':
    _test_get_graph_embedding()
    _test_get_mask_to_process()

    bs = True
    algo = ['BFS', 'parallel_coloring']
    processor = models.AlgorithmProcessor(_DIM_LATENT, bias=bs, use_gru=False).to(_DEVICE)
    load_algorithms_and_datasets(algo, processor, {'split': 'train', 'generator': 'ER', 'num_nodes': 20}, bias=bs)
    optimizer = optim.Adam(processor.parameters(),
                           lr=get_hyperparameters()[f'lr'],
                           weight_decay=get_hyperparameters()['weight_decay'])
    pprint(processor.state_dict().keys())
    print(processor)
    for epoch in range(2000):
        processor.train()
        processor.load_split('train')
        iterate_over(processor, optimizer=optimizer, epoch=epoch)
        if (epoch+1) % 1 == 0:
            processor.eval()
            print("EPOCH", epoch)
            for spl in ['val']:
                processor.load_split(spl)
                iterate_over(processor, epoch=epoch)
                for name, algorithm in processor.algorithms.items():
                    print("algo=", name)
                    pprint(algorithm.get_losses_dict(validation=True))
                    pprint(algorithm.get_validation_accuracies())
                exit(0)

    processor.eval()
    processor.load_split('test')
    iterate_over(processor, extract_formulas=True, epoch=0)
    iterate_over(processor, apply_formulas=True, epoch=0)
    for algorithm in processor.algorithms.values():
        print("EXPLANATIONS", algorithm.explanations)
        pprint(algorithm.get_validation_accuracies())
    exit(0)

    for algorithm in processor.algorithms.values():
        print(algorithm.get_validation_accuracies())
