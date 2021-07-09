import torch
from torch import nn


class Masking(nn.Module):
    def __init__(self, in_features, out_features):
        super(Masking, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, h, z):
        x = torch.cat([h, z], dim=1)
        self.h = self.linear(x)
        return self.h

