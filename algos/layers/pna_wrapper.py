import torch
import torch_geometric

from layers.pna.layer import PNALayer
from flow_datasets import SingleIterationDataset
from hyperparameters import get_hyperparameters

class PNAWrapper(PNALayer):
    def __init__(self, in_channels, edge_features, out_channels, dataset_class, bias=False, towers=1, divide_input=False):
        dataset = dataset_class('./all_iter', split='train', less_wired=True, device='cpu')
        dlist = [torch_geometric.utils.degree(dataset[i].edge_index[0].to(get_hyperparameters()["device"]))
                 for i in range(len(dataset))]
        avg_d = dict(lin=sum([torch.mean(D) for D in dlist]) / len(dlist),
                     exp=sum([torch.mean(torch.exp(torch.div(1, D)) - 1) for D in dlist]) / len(dlist),
                     log=sum([torch.mean(torch.log(D + 1)) for D in dlist]) / len(dlist))
        super(PNAWrapper, self).__init__(in_channels, edge_features, out_channels, get_hyperparameters()["pna_aggregators"].split(), get_hyperparameters()["pna_scalers"].split(), avg_d, towers=towers, divide_input=divide_input)
    def message(self, x_i, x_j, edge_attr):
        if self.divide_input:
            return torch.cat(
                   [tower.pretrans(torch.cat(
                       [x_i[:, n_tower * self.input_tower: (n_tower + 1) * self.input_tower],
                       x_j[:, n_tower * self.input_tower: (n_tower + 1) * self.input_tower],
                       edge_attr[:, n_tower * self.input_tower: (n_tower + 1) * self.input_tower]], dim=-1))
                    for n_tower, tower in enumerate(self.towers)], dim=-1)

        return torch.cat([tower.pretrans(torch.cat((x_i, x_j, edge_attr), dim=-1)) for tower in self.towers], dim=-1)

    def forward(self, x, edge_attr, edge_index):
        x = x.unsqueeze(0)
        adj_features = torch_geometric.utils.to_dense_adj(edge_index, edge_attr=edge_attr)
        adj = torch_geometric.utils.to_dense_adj(edge_index, edge_attr=None)
        result = super().forward(x, adj, adj_features).squeeze()
        return result
