import torch
import torch.nn as nn
import torch_geometric

class PrediNet(nn.Module):
    def __init__(self,
                 latent_dim,
                 num_heads,
                 key_size,
                 relations,
                 flatten_pooling=torch_geometric.nn.glob.global_max_pool):
        super(PrediNet, self).__init__()
        self.latent_dim = latent_dim
        self.num_heads = num_heads
        self.key_size = key_size
        self.get_keys = nn.Linear(latent_dim, key_size, bias=False)
        self.flatten_pooling = flatten_pooling

        self.get_Q = nn.ModuleList()
        for i in range(2):
            self.get_Q.append(nn.Linear(latent_dim, num_heads*key_size, bias=False))

        self.embed_entities = nn.Linear(latent_dim, relations, bias=False)
        self.output = nn.Sequential(nn.Linear(num_heads * relations, latent_dim),
                                    nn.LeakyReLU())

    def forward(self, inp, batch_ids):
        batch_size = len(torch.unique(batch_ids))
        # inp shape is (batch_size*num_nodes, latent_dim)

        assert not torch.isnan(inp).any()
        inp_flatten = self.flatten_pooling(inp, batch_ids)
        assert not torch.isnan(inp_flatten).any()
        inp = inp.reshape(batch_size, -1, self.latent_dim)
        assert not torch.isnan(inp).any()
        inp_tiled = inp.unsqueeze(1).repeat(1, self.num_heads, 1, 1)
        assert not torch.isnan(inp_tiled).any()

        keys = self.get_keys(inp)
        assert not torch.isnan(keys).any()
        keys_T = keys.unsqueeze(1).repeat(1, self.num_heads, 1, 1).transpose(2, 3)
        assert not torch.isnan(keys_T).any()
        embeddings = []
        for i in range(2):
            q_i = self.get_Q[i](inp_flatten)
            assert not torch.isnan(q_i).any()
            q_i = q_i.reshape(batch_size, self.num_heads, self.key_size)
            assert not torch.isnan(q_i).any()
            q_i = q_i.unsqueeze(2)
            assert not torch.isnan(q_i).any()
            qkmul = torch.matmul(q_i, keys_T)
            assert not torch.isnan(qkmul).any()
            att_i = torch.softmax(qkmul, dim=-1)
            if torch.isnan(att_i).any():
                print(inp[0])
                print()
                print(q_i)
                print()
                print(keys_T)
                print()
                print(qkmul)
            assert not torch.isnan(att_i).any()
            feature_i = torch.squeeze(torch.matmul(att_i, inp_tiled), 2)
            assert not torch.isnan(feature_i).any()
            emb_i = self.embed_entities(feature_i)
            assert not torch.isnan(emb_i).any()
            embeddings.append(emb_i)

        dx = embeddings[0] - embeddings[1]
        assert not torch.isnan(dx).any()
        dx = dx.reshape(batch_size, -1)
        assert not torch.isnan(dx).any()
        assert not torch.isnan(self.output(dx)).any()
        return self.output(dx)


if __name__ == '__main__':
    pn = PrediNet(1, 5, 3, 4)
    inp = torch.tensor([[-1], [-2], [1], [2]]).float()
    batch_ids = torch.tensor([0, 0, 1, 1]).long()
    print("OUT", pn(inp, batch_ids, batch_size=2).shape)
    pass
