import torch
from torch import nn
import torch_geometric as pyg

from algos.mst.models.masking import Masking
from algos.mst.models.nee import NeuralExecutionEngine
from algos.mst.models.pointer import Pointer


class PointerGraphNetwork(nn.Module):
    def __init__(self, encoder_in_features, encoder_out_features, processor_out_channels, device):
        super(PointerGraphNetwork, self).__init__()
        self.nee = NeuralExecutionEngine(encoder_in_features, encoder_out_features, processor_out_channels)
        self.masking = Masking(encoder_out_features+processor_out_channels, 1)
        self.pointer = Pointer(processor_out_channels)
        self.device = device

    def forward(self, x, edge_index, ids, n_nodes, y_pointer_old, true_masking):
        y_nee = self.nee(x, edge_index, ids)
        y_masking = self.masking(self.nee.encoder_pred, self.nee.processor_pred).squeeze(-1)
        block = torch.ones((n_nodes, n_nodes)).bool()
        batch_size = len(x) // n_nodes
        block_list = [block] * batch_size
        block_mask = torch.block_diag(*block_list).to(self.device)
        y_alphas = self.pointer(self.nee.processor_pred, block_mask)
        masking = true_masking if self.training else y_masking
        y_pointer = self.update_pointers(masking, y_alphas, y_pointer_old, block_mask)
        y_pointer_symmetric = self.symmetrize_pointers(y_pointer)
        return y_nee, y_masking, y_alphas, y_pointer, y_pointer_symmetric

    def update_pointers(self, y_masking, y_alphas, y_pointer_old, block_mask):
        mask = y_masking > 0
        argmax_id = torch.argmax(y_alphas, dim=1)
        y_pointer = mask * y_pointer_old + ~mask * argmax_id
        return y_pointer

    def symmetrize_pointers(self, y_pointer):
        y_pointer_symmetric = torch.vstack([torch.arange(len(y_pointer)).to(self.device), y_pointer.to(torch.long)]).to(self.device)
        y_pointer_symmetric = pyg.utils.to_undirected(y_pointer_symmetric.to(torch.long))
        return y_pointer_symmetric
