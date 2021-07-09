import torch
import torch.nn as nn
import torch_geometric.nn as nng

class MPNN(nng.MessagePassing):
    def __init__(self, in_channels, out_channels, aggr='max', bias=False, flow='source_to_target', use_gru=False):
        super(MPNN, self).__init__(aggr=aggr, flow=flow)
        self.M = nn.Sequential(nn.Linear(2*in_channels, out_channels, bias=bias),
                               nn.LeakyReLU(),
                               nn.Linear(out_channels, out_channels, bias=bias),
                               nn.LeakyReLU()
                               )
        self.U = nn.Sequential(nn.Linear(2*in_channels, out_channels, bias=bias),
                               nn.LeakyReLU())
        self.use_gru = use_gru
        if use_gru:
            self.gru = nn.GRUCell(out_channels, out_channels, bias=bias)
        self.out_channels = out_channels
    def forward(self, x, edge_index, hidden):
        hidden = self.propagate(edge_index, x=x, hidden=hidden)
        if not self.training:
            hidden = torch.clamp(hidden, -1e9, 1e9)
        return hidden
    def message(self, x_i, x_j):
        return self.M(torch.cat((x_i, x_j), dim=1))
    def update(self, aggr_out, x, hidden):
        if self.use_gru:
            hidden = self.gru(self.U(torch.cat((x, aggr_out), dim=1)), hidden)
        else:
            hidden = self.U(torch.cat((x, aggr_out), dim=1))
        return hidden
