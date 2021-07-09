import torch
from torch import nn
import torch_geometric as pyg


class Processor(pyg.nn.MessagePassing):
    def __init__(self, in_channels, out_channels, aggr='max', bias=False, flow='source_to_target', use_gru=True):
        super(Processor, self).__init__(aggr=aggr, flow=flow)
        self.M = nn.Sequential(nn.Linear(2 * in_channels, out_channels, bias=bias),
                               nn.LeakyReLU(),
                               nn.Linear(out_channels, out_channels, bias=bias),
                               nn.LeakyReLU()
                               )
        self.U = nn.Sequential(nn.Linear(in_channels + out_channels, out_channels, bias=bias),
                               nn.LeakyReLU())
        self.use_gru = use_gru
        if use_gru:
            self.gru = nn.GRUCell(out_channels, out_channels, bias=bias)
        self.out_channels = out_channels

    def forward(self, x, edge_index, hidden=None):
        return self.propagate(edge_index, x=x, hidden=hidden)

    def message(self, edge_index, x_i, x_j):
        return self.M(torch.cat((x_i, x_j), dim=1))

    def update(self, aggr_out, x, hidden):
        if self.use_gru:
            hidden = self.gru(self.U(torch.cat((x, aggr_out), dim=1)), hidden)
        else:
            hidden = self.U(torch.cat((x, aggr_out), dim=1))
        return hidden
