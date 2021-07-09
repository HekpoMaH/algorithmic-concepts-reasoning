import torch.nn as nn

class DecoderNetwork(nn.Module):
    def __init__(self, in_features, out_features, bias=False):
        super(DecoderNetwork, self).__init__()
        self.output_net = nn.Sequential(nn.Linear(in_features, out_features, bias=bias))

    def forward(self, x):
        dist = self.output_net(x)
        return dist
