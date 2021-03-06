import torch
def get_hyperparameters():
    return {
        'dim_latent': 32,
        'dim_latent_BFS': 16,
        'dim_latent_parallel_coloring': 64,
        'dim_nodes_AugmentingPath': 1,
        'dim_nodes_AugmentingPath_integers': 1,
        'dim_nodes_BFS': 2,
        'dim_nodes_parallel_coloring': 14,
        'dim_concept_BFS': 2,
        'dim_concept_parallel_coloring': 7,
        'dim_target': 2,
        'dim_target_BFS': 1,
        'dim_target_parallel_coloring': 6,
        'dim_edges': 2,
        'dim_edges_BFS': 1,
        'dim_edges_parallel_coloring': 1,
        'dim_bits': 8,
        'pna_aggregators': 'mean min max',
        'pna_scalers': 'identity amplification attenuation',
        'batch_size': 32,
        'walk_length': 5,
        'max_threshold': 10,
        'patience_limit': 10,
        'growth_rate_sigmoid': 0.0020,
        'sigmoid_offset': -300,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'calculate_termination_statistics': False,
        'lr': 0.0010000000000000000,
        'lr_BFS': 0.00050000000000,
        'lr_parallel_coloring': 0.0005000000,
        'weight_decay': 0.00000,
        'clip_grad_norm': 5,
        'clip_grad_value': 5,
        'bias': True,
        'seed': 47,
        'generators': ['ER', 'ladder', 'grid', 'tree', 'BA', 'caveman', '4-community'],
    }
