import copy
from pprint import pprint
from typing import List

import deep_logic
import torch
from torch import nn
from torch.nn.functional import nll_loss, cross_entropy, binary_cross_entropy_with_logits
from torch.optim import AdamW, Adam
import pytorch_lightning as pl
import torch_geometric as pyg

from algos.mst.datasets import generate_dataset
from algos.mst.models.minfinder import MinFinder
from algos.mst.models.pgn import PointerGraphNetwork
from algos.mst.models.explainer import Explainer


class KruskalNetwork(nn.Module):
    def __init__(self, encoder_in_features, encoder_out_features, processor_out_channels,
                 bits, embedding_size, dim_feedforward, nhead, n_concepts, n_outputs, device, no_use_concepts):
        assert nhead == 1 # NOTE NEE paper uses only 1 head
        super(KruskalNetwork, self).__init__()
        self.pgn = PointerGraphNetwork(encoder_in_features, encoder_out_features, processor_out_channels, device)
        self.minfinder = MinFinder(bits, embedding_size, dim_feedforward, nhead, no_use_concepts)
        self.explainer = Explainer(processor_out_channels, embedding_size, n_concepts, n_outputs, no_use_concepts)
        self.no_use_concepts = no_use_concepts

    def find_mins(self, edge_features_binary, last_edge_in_mst, bincounts, nodes_in_batch, edge_index):
        y_minfinder = self.minfinder(edge_features_binary, last_edge_in_mst)
        pointer_indxs = y_minfinder.argmax(-1)
        bincounts = bincounts.cumsum(-1)
        pointer_indxs[1:] += bincounts[:-1]
        inp_ptrs = torch.zeros((nodes_in_batch),)
        inp_ptrs[edge_index[:, pointer_indxs].view(-1)] = 1
        return y_minfinder, inp_ptrs

    def forward(self, edge_features_binary, last_edge_in_mst, node_features, edge_index, bincounts, edge_index_t, ids,
                n_nodes, y_pointer_old, mask_fake_edges, mask_graphs, mask_edge_index, true_mu):
        y_nee, y_masking, y_alphas, y_pointer, y_pointer_symmetric = self.pgn(node_features, edge_index_t, ids, n_nodes,
                                                                              y_pointer_old, true_mu)
        y_explainer = self.explainer(self.pgn.nee.processor_pred, self.minfinder.h, edge_index, mask_fake_edges,
                                     mask_graphs, mask_edge_index)
        return y_nee, y_masking, y_alphas, y_pointer, y_pointer_symmetric, self.explainer.concepts, y_explainer


class KruskalNetworkPL(pl.LightningModule):
    def __init__(self, encoder_in_features, encoder_out_features, processor_out_channels,
                 bits, embedding_size, dim_feedforward, nhead, n_concepts, n_outputs, n_nodes, max_n_edges,
                 optimizer: str = 'adamw', lr: float = 0.001, device: str = 'cuda', simplify: bool = False,
                 topk_explanations: int = 3, concept_names: List[str] = None, no_use_concepts=False):
        super(KruskalNetworkPL, self).__init__()
        self.model = KruskalNetwork(encoder_in_features, encoder_out_features, processor_out_channels,
                                    bits, embedding_size, dim_feedforward, nhead, n_concepts, n_outputs, device, no_use_concepts)
        self.no_use_concepts = no_use_concepts
        self.bits = bits
        self.processor_out_channels = processor_out_channels
        self.n_nodes = n_nodes
        self.max_n_edges = max_n_edges
        self.optmizer = optimizer
        self.lr = lr
        self.explanations = None
        self.use_formulas = False
        self.simplify = simplify
        self.topk_explanations = 5000
        self.concept_names = concept_names
        self.save_hyperparameters()

    def forward(self, batch):
        unnecessary_steps = (batch.edge_features == -1).sum(0).squeeze() == batch.num_graphs
        def remove_unnecessary_steps(batch, unnecessary_steps):
            batch.asymmetric_pointers_index = batch.asymmetric_pointers_index[~unnecessary_steps]
            batch.edge_concepts = batch.edge_concepts[:, ~unnecessary_steps, :, :]
            batch.edge_concepts = batch.edge_concepts[:, :, ~unnecessary_steps, :]
            batch.edge_features = batch.edge_features[:, :, ~unnecessary_steps, :]
            batch.edge_features_binary = batch.edge_features_binary[:, ~unnecessary_steps, :, :]
            batch.edge_index = batch.edge_index[:, ~unnecessary_steps, :]
            batch.edge_targets = batch.edge_targets[:, ~unnecessary_steps, :, :]
            batch.edge_targets = batch.edge_targets[:, :, ~unnecessary_steps, :]
            batch.nodes_in_different_sets = batch.nodes_in_different_sets[:, :, ~unnecessary_steps]
            batch.nodes_not_in_path_to_roots = batch.nodes_not_in_path_to_roots[:, ~unnecessary_steps, :]
            batch.x = batch.x[:, ~unnecessary_steps, :]
            return batch
        last_hidden = torch.zeros((batch.x.shape[0], self.processor_out_channels)).to(self.device)
        last_y_pointer = batch.asymmetric_pointers_index[0]
        last_edge_index = pyg.utils.to_undirected(batch.edge_index[:, 0])
        last_concepts = batch.edge_concepts[:, :, 0]
        last_explainer = batch.edge_targets[:, :, 0]
        last_node_features = torch.cat([batch.x[:, 0], last_hidden], dim=1).to(self.device)
        last_edge_in_mst = batch.edge_concepts[:, :, 0, 2]
        assert (last_edge_in_mst <= 0).all()
        edge_features_binary = batch.edge_features_binary.squeeze(2)

        actual = {'y_minfinder': [], 'y_nee': [], 'y_masking': [], 'y_asymmetric_pointers': [], 'y_concepts': [],
                  'y_explainer': [], 'y_concepts_minedge': [], 'y_explainer_minedge': []}
        predictions = {'y_minfinder': [], 'y_nee': [], 'y_masking': [], 'y_alphas': [], 'y_concepts': [],
                       'y_explainer': [], 'y_nextmask': [], 'y_concepts_minedge': [], 'y_explainer_minedge': []}

        last_actual = {'y_minfinder': [], 'y_nee': [], 'y_masking': [], 'y_asymmetric_pointers': [], 'y_concepts': [],
                  'y_explainer': [], 'y_concepts_minedge': [], 'y_explainer_minedge': []}
        last_predictions = {'y_minfinder': [], 'y_nee': [], 'y_masking': [], 'y_alphas': [], 'y_concepts': [],
                       'y_explainer': [], 'y_nextmask': [], 'y_concepts_minedge': [], 'y_explainer_minedge': []}

        elim = 5999999
        if self.use_formulas:
            assert self.explanations is not None, 'you did not give explanations'
        bincounts = torch.bincount(batch.edge_nodes_index_batch)
        self.max_n_edges = len(batch.edge_features[0, 0])-1
        for time in range(1, self.max_n_edges+1):
            self.model.minfinder.elim = elim
            self.model.minfinder.current_epoch = self.current_epoch

            # mask out graphs which do not need supervision anymore (all edges have already been checked)
            mask_graphs = (batch.edge_features[:, 0, time-1, :] == -1).squeeze(-1)
            mask_n_graphs = (batch.edge_features[:, 0, time, :] == -1).squeeze(-1)
            last_mask_graphs = (~mask_graphs & mask_n_graphs)
            if time == self.max_n_edges:
                last_mask_graphs = ~mask_graphs

            sumtaken = (batch.edge_concepts[:, :, time-1, 2] == 1).sum()


            # if all graphs have been checked, then exit and go to the next batch!
            if mask_graphs.all():
                break

            if self.training:
                pass
                last_edge_in_mst = batch.edge_targets[:, :, time-1, 0]
                last_edge_index = pyg.utils.to_undirected(batch.edge_index[:, time-1])
                last_concepts = batch.edge_concepts[:, :, time-1]

            mask_nodes = mask_graphs[batch.batch]
            last_mask_nodes = last_mask_graphs[batch.batch]
            mask_fake_edges = (batch.edge_concepts[~mask_graphs, :, time] == -1).all(dim=-1)
            last_mask_fake_edges = (batch.edge_concepts[last_mask_graphs, :, time] == -1).all(dim=-1)
            mask_all_fake_edges = (batch.edge_concepts[:, :, time] == -1).all(dim=-1)
            mask_edge_index = mask_nodes[batch.edge_nodes_index[0]]
            mask_edges_visited = last_concepts[:, :, 0] == 1

            if mask_edges_visited.any():
                edge_features_binary = edge_features_binary.clone()   # https://discuss.pytorch.org/t/assignment-using-a-byte-mask/9385
                edge_features_binary[mask_edges_visited] = torch.full_like(edge_features_binary[mask_edges_visited], -1)

            y_minfinder, inp_ptrs = self.model.find_mins(edge_features_binary, last_edge_in_mst, bincounts, len(batch.batch), batch.edge_nodes_index)

            if self.current_epoch >= elim:
                print('TIME', time)
                pow2 = torch.exp2(torch.arange(self.bits).float().cuda()).flip(0)
                print("EDGEW", (edge_features_binary*pow2).sum(dim=-1)[0])
                print("mfe", mask_fake_edges[0])
                print("mev", mask_edges_visited[0])
                print('predicted min', y_minfinder[0])
                print('actual min', batch.edge_features[~mask_graphs, :, time-1, :].squeeze(-1).squeeze(-1)[0])

            # get predictions
            last_node_features = torch.cat([batch.x[:, time], last_hidden], dim=1).to(self.device)

            if not self.training:
                last_node_features[:, 1] = inp_ptrs

            y_preds = self.model(edge_features_binary, last_edge_in_mst, last_node_features, batch.edge_nodes_index, bincounts, last_edge_index, batch.batch,
                    self.n_nodes, last_y_pointer, mask_fake_edges, mask_graphs, mask_edge_index, batch.nodes_not_in_path_to_roots[:, time].squeeze(-1))

            y_nee, y_masking, y_alphas, y_pointer, y_pointer_symmetric, y_concepts, y_explainer = y_preds
            if self.no_use_concepts:
                y_concepts[:, 0] = self.model.minfinder.new_mask[~mask_graphs][~mask_fake_edges].squeeze(-1)
            if self.use_formulas:
                assert not self.no_use_concepts, 'this model does not have concepts'
                if not self.explanations[1]:
                    y_explainer = torch.full_like(y_explainer, -1e3)
                else:
                    _, y_explainer = deep_logic.logic.test_explanation(
                            self.explanations[1] if self.explanations[1] else '(False)',
                            1,
                            y_concepts,
                            (y_concepts > 0).long(),
                            concept_names=self.concept_names,
                            give_local=True)
                    y_explainer = torch.where(y_explainer.unsqueeze(-1), 1e3, -1e3)

            lc = last_concepts[~mask_graphs]
            lc[~mask_fake_edges] = (y_concepts > 0).float()
            last_concepts[~mask_graphs] = lc

            le = last_explainer[~mask_graphs]
            le[~mask_fake_edges] = (y_explainer > 0).float()
            last_explainer[~mask_graphs] = le

            last_edge_index = y_pointer_symmetric
            leim = last_edge_in_mst[~mask_graphs]
            leim[~mask_fake_edges] = (y_explainer.squeeze(-1) > 0).float()
            last_edge_in_mst[~mask_graphs] = leim
            last_hidden = self.model.pgn.nee.processor_pred
            last_y_pointer = y_pointer

            if self.current_epoch >= elim:
                print("predicted concepts", last_concepts[0][:, 0])
                print('actual concepts', batch.edge_concepts[~mask_graphs][:, :, time][0][:, 0])
                input()

            actualminidx = batch.edge_features[~mask_graphs, :, time-1, :].squeeze(-1).squeeze(-1).clone()
            actualminidx[1:] += (bincounts[~mask_graphs].cumsum(-1)[:-1])
            actualminidx = actualminidx.long()

            actual['y_minfinder'].append(batch.edge_features[~mask_graphs, :, time-1, :].squeeze(-1).squeeze(-1))
            actual['y_nee'].append(batch.nodes_in_different_sets[~mask_graphs, :, time].squeeze(-1))
            actual['y_masking'].append(batch.nodes_not_in_path_to_roots[~mask_nodes, time].squeeze(-1))
            actual['y_asymmetric_pointers'].append(batch.asymmetric_pointers_index[time, ~mask_nodes])
            actual['y_concepts'].append(batch.edge_concepts[~mask_graphs][:, :, time][~mask_fake_edges])
            self.epoch_actual['y_concepts'].append(batch.edge_concepts[~mask_graphs][:, :, time][~mask_fake_edges])
            actual['y_concepts_minedge'].append(batch.edge_concepts[~mask_graphs][:, :, time][~mask_fake_edges][actualminidx])
            actual['y_explainer'].append(batch.edge_targets[~mask_graphs][~mask_fake_edges, :, time].view(-1))
            self.epoch_actual['y_explainer'].append(batch.edge_targets[~mask_graphs][~mask_fake_edges, :, time].view(-1))
            actual['y_explainer_minedge'].append(batch.edge_targets[~mask_graphs][~mask_fake_edges, :, time][actualminidx].view(-1))

            predictions['y_minfinder'].append(y_minfinder[~mask_graphs].squeeze(-1))
            predictions['y_nee'].append(y_nee[~mask_graphs])
            predictions['y_masking'].append(y_masking[~mask_nodes])
            predictions['y_alphas'].append(y_alphas[~mask_nodes])
            predictions['y_concepts'].append(y_concepts)
            self.epoch_predictions['y_concepts'].append(y_concepts)
            predictions['y_concepts_minedge'].append(y_concepts[actualminidx])
            predictions['y_explainer'].append(y_explainer.view(-1))
            self.epoch_predictions['y_explainer'].append(y_explainer.view(-1))
            predictions['y_explainer_minedge'].append(y_explainer[actualminidx].view(-1))

            if (last_mask_graphs).any():
                last_actualminidx = batch.edge_features[last_mask_graphs, :, time-1, :].squeeze(-1).squeeze(-1).clone()
                last_actualminidx[1:] += (bincounts[last_mask_graphs].cumsum(-1)[:-1])
                last_actualminidx = last_actualminidx.long()

                last_mask_edge_index = last_mask_nodes[batch.edge_nodes_index[0]]
                last_y_concepts = last_concepts[last_mask_graphs][~last_mask_fake_edges]
                last_y_explainer = last_explainer[last_mask_graphs][~last_mask_fake_edges]

                last_predictions['y_minfinder'].append(y_minfinder[last_mask_graphs].squeeze(-1))
                last_predictions['y_nee'].append(y_nee[last_mask_graphs])
                last_predictions['y_masking'].append(y_masking[last_mask_nodes])
                last_predictions['y_alphas'].append(y_alphas[last_mask_nodes])
                last_predictions['y_concepts'].append(last_y_concepts)
                last_predictions['y_concepts_minedge'].append(last_y_concepts[last_actualminidx])
                last_predictions['y_explainer'].append(last_y_explainer.view(-1))
                last_predictions['y_explainer_minedge'].append(last_y_explainer[last_actualminidx].view(-1))

                last_actual['y_minfinder'].append(batch.edge_features[last_mask_graphs, :, time-1, :].squeeze(-1).squeeze(-1))
                last_actual['y_nee'].append(batch.nodes_in_different_sets[last_mask_graphs, :, time].squeeze(-1))
                last_actual['y_masking'].append(batch.nodes_not_in_path_to_roots[last_mask_nodes, time].squeeze(-1))
                last_actual['y_asymmetric_pointers'].append(batch.asymmetric_pointers_index[time, last_mask_nodes])
                last_actual['y_concepts'].append(batch.edge_concepts[last_mask_graphs][:, :, time][~last_mask_fake_edges])
                last_actual['y_concepts_minedge'].append(batch.edge_concepts[last_mask_graphs][:, :, time][~last_mask_fake_edges][last_actualminidx])
                last_actual['y_explainer'].append(batch.edge_targets[last_mask_graphs][~last_mask_fake_edges, :, time].view(-1))
                last_actual['y_explainer_minedge'].append(batch.edge_targets[last_mask_graphs][~last_mask_fake_edges, :, time][last_actualminidx].view(-1))

        return predictions, actual, last_predictions, last_actual

    def compute_loss(self, predictions, actual):
        concept_preds = torch.cat(predictions['y_concepts'])
        concepts_actual = torch.cat(actual['y_concepts'])
        if self.no_use_concepts:
            concept_preds = concept_preds[:, 0]
            concepts_actual = concepts_actual[:, 0]

        loss_dict = {
            'y_minfinder': 1*nll_loss(torch.log(torch.cat(predictions['y_minfinder'])+1e-9), torch.cat(actual['y_minfinder']).long()),
            'y_nee': 1*binary_cross_entropy_with_logits(torch.cat(predictions['y_nee']), torch.cat(actual['y_nee'])),
            'y_masking': 1*10*binary_cross_entropy_with_logits(torch.cat(predictions['y_masking']), torch.cat(actual['y_masking']), pos_weight=torch.tensor(0.1)),
            'y_alphas': 1*cross_entropy(torch.cat(predictions['y_alphas']), torch.cat(actual['y_asymmetric_pointers']).long()),
            'y_concepts': 1*binary_cross_entropy_with_logits(concept_preds, concepts_actual),
            'y_explainer': 1*binary_cross_entropy_with_logits(torch.cat(predictions['y_explainer']), torch.cat(actual['y_explainer'])),
        }

        loss = sum([v for _, v in loss_dict.items()])
        if loss_dict['y_concepts'] < 0:
            print(torch.cat(predictions['y_concepts']))
            print(torch.cat(actual['y_concepts']))

            input()
        loss_np = {k: v.cpu().detach().item() if torch.is_tensor(v) else v for k, v in loss_dict.items()}
        print(f'\nLosses: {loss_np}')
        print(f'Global loss: {loss:.4f}')
        return loss

    def compute_accuracies(self, predictions, actual, last_predictions, last_actual):
        accuracies_dict = {
            'y_minfinder': torch.cat(actual['y_minfinder']).eq(torch.cat(predictions['y_minfinder']).argmax(dim=-1)).sum() / len(torch.cat(actual['y_minfinder'])),
            'y_nee': torch.cat(actual['y_nee']).eq(torch.cat(predictions['y_nee']) > 0).sum() / len(torch.cat(actual['y_nee'])),
            'y_masking': torch.cat(actual['y_masking']).eq(torch.cat(predictions['y_masking']) > 0).sum() / len(torch.cat(actual['y_masking'])),
            'y_alphas': torch.cat(actual['y_asymmetric_pointers']).eq(torch.cat(predictions['y_alphas']).argmax(dim=-1)).sum() / len(torch.cat(actual['y_asymmetric_pointers'])),
            'y_concepts': torch.cat(actual['y_concepts']).eq(torch.cat(predictions['y_concepts']) > 0).sum() / torch.prod(torch.tensor(torch.cat(actual['y_concepts']).shape)),
            'y_concepts_minedge': torch.cat(actual['y_concepts_minedge']).eq(torch.cat(predictions['y_concepts_minedge']) > 0).sum() / torch.prod(torch.tensor(torch.cat(actual['y_concepts_minedge']).shape)),
            'y_per_concepts': torch.cat(actual['y_concepts']).eq(torch.cat(predictions['y_concepts']) > 0).sum(0) / len(torch.cat(actual['y_concepts'])),
            'y_per_concepts_minedge': torch.cat(actual['y_concepts_minedge']).eq(torch.cat(predictions['y_concepts_minedge']) > 0).sum(0) / len(torch.cat(actual['y_concepts_minedge'])),
            'y_explainer': torch.cat(actual['y_explainer']).eq(torch.cat(predictions['y_explainer']) > 0).sum() / len(torch.cat(actual['y_explainer'])),
            'y_explainer_minedge': torch.cat(actual['y_explainer_minedge']).eq(torch.cat(predictions['y_explainer_minedge']) > 0).sum() / len(torch.cat(actual['y_explainer_minedge'])),

            'last_y_minfinder': torch.cat(last_actual['y_minfinder']).eq(torch.cat(last_predictions['y_minfinder']).argmax(dim=-1)).sum() / len(torch.cat(last_actual['y_minfinder'])),
            'last_y_nee': torch.cat(last_actual['y_nee']).eq(torch.cat(last_predictions['y_nee']) > 0).sum() / len(torch.cat(last_actual['y_nee'])),
            'last_y_masking': torch.cat(last_actual['y_masking']).eq(torch.cat(last_predictions['y_masking']) > 0).sum() / len(torch.cat(last_actual['y_masking'])),
            'last_y_alphas': torch.cat(last_actual['y_asymmetric_pointers']).eq(torch.cat(last_predictions['y_alphas']).argmax(dim=-1)).sum() / len(torch.cat(last_actual['y_asymmetric_pointers'])),
            'last_y_concepts': torch.cat(last_actual['y_concepts']).eq(torch.cat(last_predictions['y_concepts']) > 0).sum() / torch.prod(torch.tensor(torch.cat(last_actual['y_concepts']).shape)),
            'last_y_concepts_minedge': torch.cat(last_actual['y_concepts_minedge']).eq(torch.cat(last_predictions['y_concepts_minedge']) > 0).sum() / torch.prod(torch.tensor(torch.cat(last_actual['y_concepts_minedge']).shape)),
            'last_y_per_concepts': torch.cat(last_actual['y_concepts']).eq(torch.cat(last_predictions['y_concepts']) > 0).sum(0) / len(torch.cat(last_actual['y_concepts'])),
            'last_y_per_concepts_minedge': torch.cat(last_actual['y_concepts_minedge']).eq(torch.cat(last_predictions['y_concepts_minedge']) > 0).sum(0) / len(torch.cat(last_actual['y_concepts_minedge'])),
            'last_y_explainer': torch.cat(last_actual['y_explainer']).eq(torch.cat(last_predictions['y_explainer']) > 0).sum() / len(torch.cat(last_actual['y_explainer'])),
            'last_y_explainer_minedge': torch.cat(last_actual['y_explainer_minedge']).eq(torch.cat(last_predictions['y_explainer_minedge']) > 0).sum() / len(torch.cat(last_actual['y_explainer_minedge'])),
        }
        for k in accuracies_dict.keys():
            if len(accuracies_dict[k].shape) == 0:
                accuracies_dict[k] = accuracies_dict[k].item()
            else:
                accuracies_dict[k] = accuracies_dict[k].tolist()
        print("Accuracies: ", end='')
        pprint(accuracies_dict)

        return accuracies_dict

    def training_step(self, batch, batch_idx):
        predictions, actual, last_predictions, last_actual = self.forward(batch)
        loss = self.compute_loss(predictions, actual)
        accuracy = self.compute_accuracies(predictions, actual, last_predictions, last_actual)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_acc', accuracy['y_explainer'], on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def on_epoch_start(self):
        self.epoch_actual = {'y_minfinder': [], 'y_nee': [], 'y_masking': [], 'y_asymmetric_pointers': [], 'y_concepts': [],
                  'y_explainer': [], 'y_concepts_minedge': [], 'y_explainer_minedge': []}
        self.epoch_predictions = {'y_minfinder': [], 'y_nee': [], 'y_masking': [], 'y_alphas': [], 'y_concepts': [],
                       'y_explainer': [], 'y_nextmask': [], 'y_concepts_minedge': [], 'y_explainer_minedge': []}

    def on_epoch_end(self):
        pass

    def extract_formulas(self):
        self.explanations = {}
        for cls in range(2):
            self.explanations[cls], _ = deep_logic.logic.explain_class(model=self.model.explainer.concepts2outputs.to(self.device),
                                                               x=torch.cat(self.epoch_predictions['y_concepts']).to(torch.float).to(self.device),
                                                               y=torch.cat(self.epoch_actual['y_explainer']).to(torch.float).to(self.device),
                                                               binary=True, target_class=cls,
                                                               simplify=self.simplify,
                                                               topk_explanations=self.topk_explanations,
                                                               concept_names=self.concept_names)

    def validation_step(self, batch, batch_idx):
        predictions, actual, last_predictions, last_actual = self.forward(batch)
        loss = self.compute_loss(predictions, actual)
        accuracy = self.compute_accuracies(predictions, actual, last_predictions, last_actual)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('val_acc', accuracy['y_explainer'], on_step=True, on_epoch=True, prog_bar=True)
        self.log(f'test_last_step_acc_explainer', accuracy[f'last_y_explainer'], on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def test_step(self, batch, batch_idx):
        predictions, actual, last_predictions, last_actual = self.forward(batch)
        loss = self.compute_loss(predictions, actual)
        accuracy = self.compute_accuracies(predictions, actual, last_predictions, last_actual)

        for t, pr in zip(['mean_step', 'last_step'], ['', 'last_']):
            self.log(f'test_{t}_acc_explainer', accuracy[f'{pr}y_explainer'], on_step=True, on_epoch=True, prog_bar=True)
            self.log(f'test_{t}_acc_minfinder', accuracy[f'{pr}y_minfinder'], on_step=True, on_epoch=True, prog_bar=True)
            self.log(f'test_{t}_acc_concepts',  accuracy[f'{pr}y_concepts'], on_step=True, on_epoch=True, prog_bar=True)
            self.log(f'test_{t}_acc_explainer_minedge', accuracy[f'{pr}y_explainer_minedge'], on_step=True, on_epoch=True, prog_bar=False)
            self.log(f'test_{t}_acc_concepts_minedge', accuracy[f'{pr}y_concepts_minedge'], on_step=True, on_epoch=True, prog_bar=False)
            for i in range(3):
                self.log(f'test_{t}_acc_concepts_{i}', accuracy[f'{pr}y_per_concepts'][i], on_step=True, on_epoch=True, prog_bar=False)
                self.log(f'test_{t}_acc_concepts_minedge_{i}', accuracy[f'{pr}y_per_concepts_minedge'][i], on_step=True, on_epoch=True, prog_bar=False)

        return loss

    def configure_optimizers(self):
        if self.optmizer == 'adamw':
            return Adam(self.model.parameters(), lr=self.lr)
