from torch import nn

class Pointer(nn.Module):
    def __init__(self, in_features):
        super(Pointer, self).__init__()
        self.query = nn.Linear(in_features, in_features)
        self.key = nn.Linear(in_features, in_features)

    def forward(self, x, mask):
        self.Q = self.query(x)
        self.K = self.key(x)
        self.A = self.Q.matmul(self.K.T)
        self.A[~mask] = -1000000
        return self.A
