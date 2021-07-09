import torch
import torch.nn as nn
import torch_geometric
from hyperparameters import get_hyperparameters

class GAT(torch_geometric.nn.GATConv):
    def __init__(self, in_channels, edge_features, out_channels, heads=1,  concat=False, bias=False):
        super(GAT, self).__init__(in_channels, out_channels, heads=heads, concat=concat, flow='source_to_target')
        self.pretrans = nn.Linear(in_channels, out_channels, bias=bias)
        self.att = nn.Parameter(torch.Tensor(2*in_channels+edge_features, 1))
        nn.init.xavier_uniform_(self.att)
        self.gru = nn.GRUCell(out_channels, out_channels, bias=bias)

    def zero_hidden(self, num_nodes):
        self.hidden = torch.zeros(num_nodes, self.out_channels).to(get_hyperparameters()["device"])

    def forward(self, x, edge_attr, edge_index):
        size = x.size(0)
        return self.propagate(edge_index, x=x, edge_attr=edge_attr, size=size)

    def message(self, x_i, x_j, edge_index, edge_attr, size_i):
        cat = torch.cat([x_i, x_j, edge_attr], dim=-1)
        alpha = torch.matmul(cat, self.att)
        alpha = torch_geometric.utils.softmax(alpha, edge_index[1] if self.flow == 'source_to_target' else edge_index[0], size_i).sum(dim=-1)
        x_j = self.pretrans(x_j) # Not as effective but reduces API overhead
        return x_j*(alpha.unsqueeze(-1))

    def update(self, aggr_out):
        return self.gru(nn.functional.leaky_relu(aggr_out), self.hidden)
