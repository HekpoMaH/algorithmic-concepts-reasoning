import torch
from torch import nn
from torch_scatter import scatter

from algos.mst.datasets import generate_dataset
from algos.mst.models.decoder import Decoder
from algos.mst.models.encoder import Encoder
from algos.mst.models.processor import Processor


class NeuralExecutionEngine(nn.Module):
    def __init__(self, encoder_in_features, encoder_out_features, processor_out_channels):
        super(NeuralExecutionEngine, self).__init__()
        self.encoder = Encoder(encoder_in_features, encoder_out_features)
        self.processor = Processor(encoder_out_features, processor_out_channels)
        self.decoder = Decoder(encoder_out_features+processor_out_channels, 1)

    def forward(self, x, edge_index, ids):
        # encode
        self.encoder_pred = self.encoder(x)

        # process
        self.processor_pred = self.processor(self.encoder_pred, edge_index)

        # permutation-invariant aggregation
        encoder_pred = scatter(self.encoder_pred, ids, dim=0, reduce="max")
        processor_pred_max = scatter(self.processor_pred, ids, dim=0, reduce="max")
        decoder_input = torch.cat([encoder_pred, processor_pred_max], dim=1)

        # decode
        y_pred = self.decoder(decoder_input).squeeze(-1)
        return y_pred
