import torch
from torch import nn
from torch_scatter import scatter

from algos.mst.datasets import generate_dataset
from algos.mst.models.encoder import Encoder
from algos.mst.models.processor import Processor


class Decoder(nn.Module):
    def __init__(self, in_features, out_features):
        super(Decoder, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        self.h = self.linear(x)
        return self.h


if __name__ == '__main__':
    n_nodes = 5
    batch_size = 5
    n_graphs = 70
    bits = 16
    nhead = bits // 2

    loader, _, _ = generate_dataset(n_graphs=batch_size * 20, batch_size=batch_size, n_nodes=n_nodes, bits=bits)
    batch = next(iter(loader))

    k = 5
    q = 3
    in_features = batch.x.shape[2] + k
    out_features = q

    # sanity check
    encoder = Encoder(in_features, out_features)
    processor = Processor(out_features, 2)
    decoder = Decoder(out_features+2, 1)
    encoder.eval()
    processor.eval()
    decoder.eval()
    for batch_size in [2, 3, 4, 5]:
        print(f'Batch size: {batch_size}')
        loader, max_n_edges, _ = generate_dataset(n_graphs=batch_size*20, batch_size=batch_size, n_nodes=n_nodes, bits=bits)
        batch = next(iter(loader))

        time = 5
        hidden_0 = torch.zeros((batch.x.shape[0], k))
        x = torch.cat([batch.x[:, time], hidden_0], dim=1)
        y = batch.nodes_in_different_sets.squeeze()[:, time]

        # encode
        encoder_pred = encoder(x).squeeze()

        # process
        edge_index = batch.edge_index.permute(1, 0, 2).reshape(2, -1, max_n_edges+1)[:, :, time].T
        processor_pred = processor(encoder_pred, edge_index).squeeze()

        # decode
        encoder_pred = scatter(encoder_pred, batch.batch, dim=0, reduce="max")
        processor_pred_max = scatter(processor_pred, batch.batch, dim=0, reduce="max")
        decoder_input = torch.cat([encoder_pred, processor_pred_max], dim=1)
        y_pred = decoder(decoder_input).squeeze()

        print(f'decoder_input: {decoder_input}')
        print(f'y_pred: {y_pred}')

    print(f'decoder_input: {decoder_input.shape}')
    print(f'y_pred: {y_pred.shape}')
