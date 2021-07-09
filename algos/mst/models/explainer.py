import torch
from torch import nn
import deep_logic

from algos.mst.datasets import generate_dataset
from algos.mst.models.encoder import Encoder
from algos.mst.models.minfinder import MinFinder
from algos.mst.models.processor import Processor


class Explainer(nn.Module):
    def __init__(self, pgn_latent_features, minfinder_features, n_concepts, n_outputs, no_use_concepts):
        super(Explainer, self).__init__()
        self.no_use_concepts = no_use_concepts
        self.minfinder_features = minfinder_features
        if not self.no_use_concepts:
            self.latent2concepts = nn.Sequential(
                nn.Linear(minfinder_features + 2 * pgn_latent_features, minfinder_features),
                nn.LeakyReLU(),
                nn.Linear(minfinder_features, n_concepts-1))
            self.latent2C1 = nn.Sequential(
                nn.Linear(2, minfinder_features),
                nn.LeakyReLU(),
                nn.Linear(minfinder_features, 1))
            self.concepts2outputs = nn.Sequential(
                deep_logic.nn.XLogic(n_concepts, minfinder_features, activation='sigmoid'),
                nn.LeakyReLU(),
                nn.Linear(minfinder_features, n_outputs),
                deep_logic.nn.XLogic(n_outputs, n_outputs, activation='identity', top=True))
        else:
            self.latent2outputs = nn.Sequential(
                nn.Linear(minfinder_features + 2 * pgn_latent_features, minfinder_features),
                nn.LeakyReLU(),
                nn.Linear(minfinder_features, n_outputs))

    def forward(self, latent_pgn, latent_minfinder, edge_index, mask_fake_edges, mask_graphs, mask_edge_index):
        latent_pgn_per_edge = torch.cat((latent_pgn[edge_index[0, ~mask_edge_index]], latent_pgn[edge_index[1, ~mask_edge_index]]), dim=-1)
        latent_minfinder_masked = latent_minfinder[~mask_graphs][~mask_fake_edges]
        x = torch.cat([latent_minfinder_masked, latent_pgn_per_edge], dim=1)
        if not self.no_use_concepts:
            c1 = self.latent2C1(x[:, :2])
            cr = self.latent2concepts(x[:, 2:])
            self.concepts = torch.cat([c1, cr], dim=-1)
            outputs = self.concepts2outputs(self.concepts)
        else:
            self.concepts = torch.zeros((len(latent_minfinder_masked), 3)).to(latent_minfinder_masked.device)
            outputs = self.latent2outputs(x[:, 2:])

        return outputs
