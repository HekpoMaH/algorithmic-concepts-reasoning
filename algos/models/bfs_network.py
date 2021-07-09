import torch
from algos.models import AlgorithmBase
from algos.hyperparameters import get_hyperparameters

torch.set_printoptions(precision=18)
_DEVICE = get_hyperparameters()['device']
_DIM_LATENT = get_hyperparameters()['dim_latent']

class BFSNetwork(AlgorithmBase):
    def __init__(self,
                 latent_features,
                 node_features,
                 concept_features,
                 output_features,
                 algo_processor,
                 dataset_class,
                 dataset_root,
                 dataset_kwargs,
                 bias=False,
                 sigmoid=False):
        assert False, "Why use that?"
        super(BFSNetwork, self).__init__(latent_features,
                                         node_features,
                                         concept_features,
                                         output_features,
                                         algo_processor,
                                         dataset_class,
                                         dataset_root,
                                         dataset_kwargs,
                                         bias=bias,
                                         sigmoid=sigmoid)

