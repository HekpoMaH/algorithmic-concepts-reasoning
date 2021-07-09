import torch
from torch import nn
from torch.nn import functional as F

from algos.mst.datasets import generate_dataset


class Encoder(nn.Module):
    def __init__(self, in_features, out_features):
        super(Encoder, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        self.h = self.linear(x)
        return F.leaky_relu(self.h)


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
    model = Encoder(in_features, out_features)
    model.eval()
    for batch_size in [2, 3, 4, 5]:
        print(f'Batch size: {batch_size}')
        loader, _, _ = generate_dataset(n_graphs=batch_size*20, batch_size=batch_size, n_nodes=n_nodes, bits=bits)
        batch = next(iter(loader))

        time = 1
        hidden_0 = torch.zeros((batch.x.shape[0], k))
        x = torch.cat([batch.x[:, time], hidden_0], dim=1)

        y_pred = model(x).squeeze()

        print(f'x: {x}')
        print(f'y_pred: {y_pred}')

    print(f'x: {x.shape}')
    print(f'y_pred: {y_pred.shape}')
