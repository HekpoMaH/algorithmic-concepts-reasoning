import torch
import torch.nn as nn
import torch_geometric.nn as nng
from torch_geometric.utils import to_dense_adj

class PredecessorNetwork(nng.MessagePassing):
    def __init__(self, in_channels, edge_feature, aggr='max', bias=False):
        super(PredecessorNetwork, self).__init__(aggr=aggr, flow='source_to_target')
        self.pred = nn.Sequential(
                                  nn.Linear(4*in_channels+edge_feature, 1, bias=bias),
                                  nn.LeakyReLU())
    def forward(self, enc, latent, edge_attr, edge_index, **kwargs):
        x = torch.cat((enc, latent), dim=1)
        gather_idx_i = edge_index[0].view(-1, 1).repeat(1, x.shape[1])
        gather_idx_j = edge_index[1].view(-1, 1).repeat(1, x.shape[1])
        if 'orig_edge_attr' in kwargs:

            mask = kwargs['orig_edge_attr'][:, 1] > 0
            gather_idx_i = gather_idx_i[mask == True]
            gather_idx_j = gather_idx_j[mask == True]
        x_i = torch.gather(x, 0, gather_idx_i)
        x_j = torch.gather(x, 0, gather_idx_j)
        if self.flow == 'source_to_target':
            x_i, x_j = x_j, x_i

        scalar = self.pred(torch.cat((x_i, x_j, edge_attr[mask == True] if 'orig_edge_attr' in kwargs else edge_attr), dim=1))

        if 'orig_edge_attr' in kwargs:
            scalars = []
            cnt = 0
            for i in range(len(edge_index[0])):
                if kwargs['orig_edge_attr'][i][1] == 0:
                    scalars.append(0)
                else:
                    scalars.append(scalar[cnt])
                    cnt += 1

            scalar = torch.tensor(scalars, requires_grad=True).cuda()
        dadj = to_dense_adj(edge_index, edge_attr=scalar).squeeze().t()

        dadj [dadj == 0.] = -1e9
        return dadj

    def forward2(self, x, edge_index, edge_attr, **kwargs):
        x_i, x_j = ([], [])
        for i, j in zip(edge_index[0], edge_index[1]):
            x_i.append(list(x[i]))
            x_j.append(list(x[j]))
        if self.flow == 'source_to_target':
            x_i, x_j = x_j, x_i
        x_i = torch.tensor(x_i, device=kwargs['device'])
        x_j = torch.tensor(x_j, device=kwargs['device'])
        scalar = self.pred(torch.cat((x_i, x_j, edge_attr), dim=1))
        dadj = to_dense_adj(edge_index, edge_attr=scalar).squeeze()
        return dadj
