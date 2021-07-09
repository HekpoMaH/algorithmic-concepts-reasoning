import torch
import torch.nn as nn

from algos.layers import MPNN

class AlgorithmProcessor(nn.Module):
    '''
    'Container' class that holds the GNN that the algorithms share and a list of algorithms to be trained/tested.
    '''
    def __init__(self, latent_features, processor_type='MPNN', bias=True, use_gru=False, prune_logic_epoch=-1):
        assert processor_type in ['MPNN']
        super(AlgorithmProcessor, self).__init__()
        self.processor = MPNN(latent_features, latent_features, bias=bias, use_gru=use_gru)
        self.algorithms = nn.ModuleDict()
        self.prune_logic_epoch = prune_logic_epoch

    def load_split(self, split, num_nodes=20, datapoints=None):
        for algorithm in self.algorithms.values():
            algorithm.dataset_kwargs['split'] = split
            algorithm.dataset_kwargs['num_nodes'] = num_nodes
            if datapoints is not None:
                algorithm.dataset_kwargs['datapoints_non_train'] = datapoints
            algorithm.load_dataset(algorithm.dataset_class,
                                   algorithm.dataset_root,
                                   algorithm.dataset_kwargs)

    def add_algorithm(self, algorithm, name):
        self.algorithms[name] = algorithm

    def update_weights(self, optimizer):
        loss = 0
        assert self.training
        for name, algorithm in self.algorithms.items():
            loss += algorithm.get_training_loss()
        # as we can try to execute a step, but all algoritmhs have finished their epoch
        # and the loss is the int 0
        # Now I made it not call this function if no algorithm was executed
        # i.e. all algorithms are done

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    def reset_all_weights(self):
        for name, W in self.named_parameters():
            # 'code' versions of the 'presumably same' architecture produce different outputs/gradients
            W.data.fill_(0.01)

    def get_sum_grad(self):
        s = 0
        for name, W in self.named_parameters():
            print(name, W.grad)
            s += W.grad.sum()
        return s

    def check_bad_grad(self, threshold=5):
        for name, W in self.named_parameters():
            if W.grad is None:
                continue
            if (torch.norm(W.grad, p=2) > threshold).any():
                print("badgrad norm", name, W.grad)
                print(torch.norm(W.grad))
                input()

            if (W.grad > threshold).any():
                print("badgrad value", name, W.grad)
                input()

if __name__ == '__main__':
    processor = AlgorithmProcessor(32, bias=False, use_gru=True)
    print(processor)
