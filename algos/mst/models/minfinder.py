import torch
from torch import nn

from algos.mst.datasets import generate_dataset
from algos.mst.models.transformers import CustomTransformerDecoderLayer, CustomTransformerEncoderLayer

def printer(module, gradInp, gradOutp):
    print(list(module.named_parameters()))
    s = 0
    mx = -float('inf')
    for gi in gradInp:
        s += torch.sum(gi)
        mx = max(mx, torch.max(torch.abs(gi)))

    print("INP")
    print(f'sum {s}, max {mx}')
    s = 0
    mx = -float('inf')
    print("OUTP")
    for go in gradOutp:
        s += torch.sum(go)
        mx = max(mx, torch.max(torch.abs(go)))
    print(f'sum {s}, max {mx}')
    print(f'gradout {gradOutp}')
    input()


class MinFinder(nn.Module):
    def __init__(self, in_features, embedding_size, dim_feedforward, nhead, no_use_concepts):
        super(MinFinder, self).__init__()
        self.embed = nn.Linear(in_features + 1, embedding_size)
        self.attention_NN = nn.Sequential(
            nn.Linear(2*embedding_size, 1))
        self.encoder_layer = CustomTransformerEncoderLayer(d_model=embedding_size, dim_feedforward=dim_feedforward, NN=self.attention_NN, nhead=nhead, dropout=0.0)
        self.decoder_layer = CustomTransformerDecoderLayer(d_model=embedding_size, dim_feedforward=dim_feedforward, NN=self.attention_NN, nhead=nhead, dropout=0.0)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, 6)
        self.decoder = nn.TransformerDecoder(self.decoder_layer, 6)
        self.no_use_concepts = no_use_concepts
        if self.no_use_concepts:
            self.masklinear = nn.Sequential(
                    nn.Linear(2, 16),
                    nn.LeakyReLU(),
                    nn.Linear(16, 1))

    def forward(self, x, last_edge_in_mst):
        src_key_padding_mask = x.sum(dim=-1) > 0
        fake_edge_mask = ~((x == 0).all(dim=-1))
        vis_mask = (x == -1).all(dim=-1)
        badrows = ((~fake_edge_mask) & vis_mask).any(dim=-1)
        fake_edge_mask[badrows, 0] = True
        vis_mask[badrows, 0] = True
        assert (src_key_padding_mask[~badrows] == (fake_edge_mask & ~vis_mask)[~badrows]).all()

        weights_and_mst = torch.cat([x, last_edge_in_mst.unsqueeze(-1)], dim=-1)

        self.embedding = self.embed(weights_and_mst)

        self.h = self.encoder(self.embedding.permute(1, 0, 2), src_key_padding_mask=(~fake_edge_mask)&vis_mask)
        zvector = torch.zeros_like(self.h[0].unsqueeze(0))
        _ = self.decoder(zvector, self.h, memory_key_padding_mask=~fake_edge_mask).permute(1, 0, 2)
        self.h = self.h.permute(1, 0, 2)

        attn = self.decoder.layers[-1].attention.permute(1, 0, 2).squeeze(0)
        attnargm = torch.argmax(attn, dim=-1)
        device = self.embed.weight.device
        attn2 = torch.zeros((self.h.shape[0], self.h.shape[1]), ).to(device)
        attn2[torch.arange(len(self.h)).to(device), attnargm] = 1
        vis_mask[badrows, 0] = False
        self.h = torch.cat((attn2.unsqueeze(-1), vis_mask.unsqueeze(-1).float(), self.h), dim=-1)
        if self.no_use_concepts:
            inpa = nn.functional.one_hot(attn.argmax(dim=-1), num_classes=len(attn[0])).float()
            cnn_inp = torch.stack((~vis_mask, inpa), dim=-1)
            self.new_mask = self.masklinear(cnn_inp)
        if self.current_epoch >= self.elim:
            print("after linear", self.new_mask[0])

        assert (attn[~fake_edge_mask] == 0).all(), attn[~fake_edge_mask]
        return attn
