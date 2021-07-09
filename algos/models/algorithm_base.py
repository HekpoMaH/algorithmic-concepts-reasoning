import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
import torch_scatter
import deep_logic
from deep_logic.utils.layer import l1_loss

import algos.utils as utils
import algos.models as models
from algos.hyperparameters import get_hyperparameters
from algos.layers import GlobalAttentionPlusCoef, PrediNet

_DEVICE = get_hyperparameters()['device']
_DIM_LATENT = get_hyperparameters()['dim_latent']

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

class AlgorithmBase(nn.Module):
    '''
    Base class for algorithm's execution. The class takes into
    account that applying the SAME algorithm to different graphs
    may take DIFFERENT number of steps. The class implementation
    circumvents this problems by masking out graphs (or 'freezing'
    to be precise) which should have stopped executing.

    It also re-calculates losses/metrics, so that they do not differ
    between using batch size of 1 or batch size of >1.
    '''

    @staticmethod
    def get_masks(train, batch, continue_logits, enforced_mask):
        '''
            mask is which nodes out of the batched disconnected graph should
            not change their output/latent state.

            mask_cp is which graphs of the batch should be frozen (mask for
            Continue Probability).

            Once a graph/node is frozen it cannot be unfreezed.

            Masking is important so we don't change testing dynamics --
            testing the same model with batch size of 1 should give the
            same results as testing with batch size of >1.

            continue logits: Logit values for the continue probability
            for each graph in the batch.

            enforced mask: Debugging tool to forcefully freeze a graph.
        '''
        if train:
            mask = torch.ones_like(batch.batch).bool()
            mask_cp = torch.ones_like(continue_logits).bool()
        else:
            mask = utils.get_mask_to_process(continue_logits, batch.batch)
            mask_cp = (continue_logits > 0.0).bool()
            if enforced_mask is not None:
                enforced_mask_ids = enforced_mask[batch.batch]
                mask &= enforced_mask_ids
                mask_cp &= enforced_mask
        return mask, mask_cp

    def load_dataset(self, dataset_class, dataset_root, dataset_kwargs):
        self.dataset_class = dataset_class
        self.dataset_root = dataset_root
        self.dataset_kwargs = dataset_kwargs
        self.dataset = dataset_class(dataset_root, self.inside_class, **dataset_kwargs)
        if hasattr(self.dataset, 'concept_pos_weights') and self.drop_last_concept:
            self.dataset.concept_pos_weights = self.dataset.concept_pos_weights[:-1]

    def __init__(self,
                 latent_features,
                 node_features,
                 concept_features,
                 output_features,
                 algo_processor,
                 dataset_class,
                 inside_class,
                 dataset_root,
                 dataset_kwargs,
                 bias=False,
                 use_TF=False,
                 use_concepts=True,
                 use_concepts_sv=True,
                 drop_last_concept=False,
                 L1_loss=False,
                 prune_logic_epoch=-1,
                 use_decision_tree=False,
                 activation_xlogic='sigmoid',
                 global_termination_pool='predinet', #'max',
                 next_step_pool=True,
                 get_attention=False,
                 use_batch_norm=False,
                 sigmoid=False):

        super(AlgorithmBase, self).__init__()
        if not isinstance(self, models.AlgorithmColoring):
            drop_last_concept = False # We don't want this for BFS
        self.node_features = node_features
        self.concept_features = concept_features - int(drop_last_concept)
        self.output_features = output_features
        self.latent_features = latent_features
        self.debug = False
        self.epoch_threshold_debug = 500
        self.use_concepts = use_concepts
        self.use_concepts_sv = use_concepts_sv
        self.drop_last_concept = drop_last_concept
        self.L1_loss = L1_loss
        self.prune_logic_epoch = prune_logic_epoch
        self.use_decision_tree = use_decision_tree
        assert not use_concepts_sv or use_concepts, 'using supervision on concepts implies using concepts'
        self.activation_xlogic = activation_xlogic
        self.global_termination_pool = global_termination_pool
        self.next_step_pool = next_step_pool
        self.processor = algo_processor.processor
        self.use_TF = use_TF
        self.get_attention = get_attention
        self.lambda_mul = 1# 0.0001
        self.concepts_unlearned_mul = 1
        self.inside_class = inside_class
        self.load_dataset(dataset_class, dataset_root, dataset_kwargs)

        self.node_encoder = nn.Sequential(
            nn.Linear(node_features + latent_features, latent_features, bias=bias),
            nn.LeakyReLU()
        )
        self.decoder_network = nn.Sequential(
            nn.Linear(2 * latent_features, latent_features, bias=bias),
            nn.LeakyReLU(),
            nn.Linear(latent_features, (concept_features-int(drop_last_concept)) if self.use_concepts else output_features, bias=bias)
        )

        self.decoder_network_term = nn.Sequential(
            nn.Linear(latent_features*2, (concept_features-int(drop_last_concept)) if self.use_concepts else output_features, bias=bias)
        )

        if self.use_concepts and sigmoid:
            self.concept_decoder = nn.Sequential(nn.Linear(concept_features-int(drop_last_concept), output_features, bias=bias))
        if self.use_concepts and not sigmoid:
            assert not use_batch_norm
            self.concept_decoder = nn.Sequential(
                nn.BatchNorm1d(concept_features-int(drop_last_concept)) if use_batch_norm else nn.Identity(),
                deep_logic.nn.XLogic(concept_features-int(drop_last_concept), latent_features, bias=bias, activation=activation_xlogic),
                nn.LeakyReLU(),
                nn.Linear(latent_features, output_features, bias=bias),
                deep_logic.nn.XLogic(output_features, output_features, bias=bias, activation='identity', top=True))

        if global_termination_pool == 'attention':
            inp_dim = 2*latent_features if self.use_concepts else latent_features
            self.global_attn = GlobalAttentionPlusCoef(
                    nn.Sequential(
                        nn.Linear(inp_dim, latent_features, bias=bias),
                        nn.LeakyReLU(),
                        nn.Linear(latent_features, 1, bias=bias)
                    ),
                    nn=None)

        if global_termination_pool == 'predinet':
            lf = latent_features
            self.predinet = PrediNet(lf, 1, lf, lf, flatten_pooling=torch_geometric.nn.glob.global_max_pool)

        self.termination_network = nn.Sequential(
            nn.BatchNorm1d(latent_features) if use_batch_norm else nn.Identity(),
            nn.Linear(latent_features, 1, bias=bias),
            )

    def inputify(self, x):
        def offset_weird_leaky_relu(x, start_leak=0.05):
            return torch.where((x+0.5) < 1, F.leaky_relu((x+0.5), start_leak), 1+5e-2*(x-0.5))

        if self.activation_xlogic == 'sigmoid':
            return torch.sigmoid(x)
        elif self.activation_xlogic == 'offset_leaky_relu':
            return F.leaky_relu(x+0.5)
        elif self.activation_xlogic == 'offset_weird_leaky_relu':
            return offset_weird_leaky_relu(x)
        elif self.activation_xlogic == 'offset_weird_leaky_relu/2':
            return offset_weird_leaky_relu(x/2)
        elif self.activation_xlogic == 'offset_weird_no_start_leak_relu':
            return offset_weird_leaky_relu(x, start_leak=0)
        assert False, 'unknown inputification'

    def get_continue_logits(self, batch_ids, latent_nodes, sth_else=None):
        if self.global_termination_pool == 'mean':
            graph_latent = torch_geometric.nn.global_mean_pool(latent_nodes, batch_ids)
        if self.global_termination_pool == 'max':
            graph_latent = torch_geometric.nn.global_max_pool(latent_nodes, batch_ids)
        if self.global_termination_pool == 'attention':
            graph_latent, coef = self.global_attn(latent_nodes, batch_ids)
            if self.get_attention:
                self.attentions[self.steps] = coef.clone().detach()
                self.per_step_latent[self.steps] = sth_else

        if self.global_termination_pool == 'predinet':
            assert not torch.isnan(latent_nodes).any()
            graph_latent = self.predinet(latent_nodes, batch_ids)

        if self.get_attention:
            self.attentions[self.steps] = latent_nodes
        continue_logits = self.termination_network(graph_latent).view(-1)
        concept_continue_logits = None
        return concept_continue_logits, continue_logits

    def zero_termination(self):
        self.true_positive = 0
        self.false_positive = 0
        self.false_negative = 0
        self.true_negative = 0

    def zero_steps(self):
        self.sum_of_processed_nodes, self.steps, self.sum_of_steps, self.cnt = 0, 0, 0, 0

    def zero_tracking_losses_and_statistics(self):
        if self.training:
            self.zero_termination()
            self.losses = {
                'total_loss_output': 0,
                'total_loss_term': 0,
                'total_loss_concepts': 0,
            }

    def zero_formulas_aggregation(self):
        self.predictions = {
            'outputs': [],
            'terminations': [],
            'concepts': [],
            'concepts_term': []
        }
        self.actual = {
            'outputs': [],
            'terminations': [],
            'concepts': [],
            'concepts_term': []
        }

    def zero_validation_stats(self):
        self.mean_step = []
        self.last_step = []
        self.concepts_mean_step = []
        self.per_concept_mean_step = []
        self.concepts_fin_mean_step = []
        self.concepts_last_step = []
        self.per_concept_last_step = []
        self.concepts_fin_last_step = []
        self.formula_preds_mean_step = []
        self.term_preds_mean_step = []
        self.term_formula_preds_mean_step = []
        self.formula_preds_last_step = []
        self.validation_sum_of_steps = 0
        self.validation_sum_of_processed_nodes = 0
        self.last_step_total = 0
        self.concepts_last_step_total = 0
        self.concepts_fin_last_step_total = 0
        self.formula_preds_last_step_total = 0
        if self.get_attention:
            self.attentions = {}
            self.per_step_latent = {}
        self.zero_termination()
        self.validation_losses = {
            'total_loss_output': 0,
            'total_loss_term': 0,
            'total_loss_concepts': 0,
        }
        self.losses = {
            'total_loss_output': 0,
            'total_loss_term': 0,
            'total_loss_concepts': 0,
        }
        self.wrong_indices = []

    @staticmethod
    def calculate_step_acc(output, output_real, batch_mask, take_total_for_classes=True):
        """ Calculates the accuracy for a givens step """
        output = output.squeeze(-1)
        correct = 0
        tot = 0
        correct = output == output_real
        if len(correct.shape) == 2 and take_total_for_classes:
            correct = correct.float().mean(dim=-1)
        _, batch_mask = torch.unique(batch_mask, return_inverse=True)
        correct_per_batch = torch_scatter.scatter(correct.float(), batch_mask, reduce='mean', dim=0)
        return correct_per_batch

    def get_output_loss(self, output_logits, target):
        return F.binary_cross_entropy_with_logits(output_logits.squeeze(-1), target, reduction='sum', pos_weight=torch.tensor(1.00))

    def aggregate_step_acc(self, batch, mask, mask_cp, y_curr, output_logits,
                           preds_with_formula, concepts_real, concepts_logits,
                           true_termination, continue_logits, termination_with_formula):

        masked_batch = batch.batch[mask]
        output_logits_masked = output_logits[mask]
        output_real_masked = y_curr[mask].float()
        if self.use_concepts:
            concepts_logits_masked = concepts_logits[mask]
            concepts_real_masked = concepts_real[mask]

        assert not torch.isnan(output_logits_masked).any(), output_logits_masked
        # if we are not training, calculate mean step accuracy
        # for outputs/logits/predictions from formulas
        if self.use_concepts_sv:
            assert len(masked_batch) > 0
            mean_concept_accs = type(self).calculate_step_acc(concepts_logits_masked > 0, concepts_real_masked, masked_batch)
            self.concepts_mean_step.extend(mean_concept_accs)
            per_concept_mean_concept_accs = type(self).calculate_step_acc(concepts_logits_masked > 0, concepts_real_masked, masked_batch, take_total_for_classes=False)
            self.per_concept_mean_step.extend(per_concept_mean_concept_accs)
        else:
            concept_correct, concept_tot = 0, 0

        if preds_with_formula is not None:
            mean_formula_accs = type(self).calculate_step_acc(type(self).get_outputs(preds_with_formula[mask]), output_real_masked, masked_batch)
            self.formula_preds_mean_step.extend(mean_formula_accs)
            term_formula_correct_accs = type(self).calculate_step_acc(termination_with_formula.float(), true_termination, torch.unique(batch.batch))
            if (term_formula_correct_accs != 1).any():
                self.wrong_flag = True
            self.term_formula_preds_mean_step.extend(term_formula_correct_accs)
        mean_accs = type(self).calculate_step_acc(type(self).get_outputs(output_logits_masked), output_real_masked, masked_batch)
        self.mean_step.extend(mean_accs)
        term_correct_accs = type(self).calculate_step_acc((continue_logits > 0).float(), true_termination, torch.unique(batch.batch))
        self.term_preds_mean_step.extend(term_correct_accs)
        if (mean_accs != 1).any() or (term_correct_accs != 1).any():
            self.wrong_flag = True


    def get_step_loss(self,
                      mask,
                      mask_cp,
                      y_curr,
                      output_logits,
                      preds_with_formula,
                      concepts_real,
                      concepts_logits,
                      concept_continue_logits,
                      true_termination,
                      continue_logits,
                      termination_with_formula,
                      compute_losses=True):

        # Take prediction (logits for outputs and concepts) and
        # target values (real for concepts and outputs) for
        # the graphs that still proceed with execution
        output_logits_masked = output_logits[mask]
        output_real_masked = y_curr[mask].float()
        if self.use_concepts:
            concepts_logits_masked = concepts_logits[mask]
            concepts_real_masked = concepts_real[mask]

        # Accumulate number of steps done. Each unfrozen graph contributes with 1 step.
        steps = sum(mask_cp.float())

        loss_output, loss_concepts, loss_term, processed_nodes = 0, 0, 0, 0

        # If we simply want to execute (e.g. when testing displaying), we drop
        # losses calculation
        if compute_losses:
            processed_nodes = len(output_real_masked)
            loss_output = self.get_output_loss(output_logits_masked, output_real_masked)

            loss_concepts = self.get_concept_loss(mask, concepts_real, concepts_logits, reduction='sum')
            pw = getattr(self.dataset, 'termination_pos_weights_AAAA', None)

            # calculate losses for termination from masked out graphs.
            # NOTE we use sum reduction as we will do the averaging later
            # (That's why we have sum of steps and sum of nodes)
            loss_term = F.binary_cross_entropy_with_logits(
                continue_logits[mask_cp],
                true_termination[mask_cp].float(),
                reduction='sum',
                pos_weight=pw)
            if get_hyperparameters()['calculate_termination_statistics']:
                self.update_termination_statistics(continue_logits[mask_cp], true_termination[mask_cp].float())


        return loss_output, loss_concepts, loss_term, processed_nodes

    def aggregate_loss_steps_and_acc(self,
                                     batch,
                                     mask,
                                     mask_cp,
                                     y_curr,
                                     output_logits,
                                     preds_with_formula,
                                     concepts_real,
                                     concepts_logits,
                                     concept_continue_logits,
                                     true_termination,
                                     continue_logits,
                                     termination_with_formula,
                                     compute_losses=True):

        loss_output, loss_concepts, loss_term, processed_nodes =\
                self.get_step_loss(
                    mask, mask_cp,
                    y_curr, output_logits, preds_with_formula,
                    concepts_real, concepts_logits,
                    concept_continue_logits,
                    true_termination, continue_logits, termination_with_formula,
                    compute_losses=compute_losses)

        self.losses['total_loss_output'] += loss_output
        self.losses['total_loss_concepts'] += loss_concepts
        self.losses['total_loss_term'] += loss_term

        if not self.training:
            self.aggregate_step_acc(batch, mask, mask_cp, y_curr, output_logits,
                                    preds_with_formula, concepts_real,
                                    concepts_logits, true_termination, continue_logits,
                                    termination_with_formula)

        steps = sum(mask_cp.float())
        self.aggregate_steps(steps, processed_nodes)

    def aggregate_steps(self, steps, processed_nodes):
        self.sum_of_steps += steps
        self.sum_of_processed_nodes += processed_nodes
        self.steps += 1
        if not self.training:
            self.validation_sum_of_processed_nodes += processed_nodes
            self.validation_sum_of_steps += steps

    def aggregate_last_step(self, batch_ids, output, real, concepts_output, concepts_real, formula_preds):
        last_step_accs = type(self).calculate_step_acc(type(self).get_outputs(output), real, batch_ids)
        if (last_step_accs != 1).any():
            self.wrong_flag = True
        self.last_step.extend(last_step_accs)
        self.last_step_total += len(last_step_accs)
        if self.use_concepts_sv:
            last_step_concept_accs = type(self).calculate_step_acc(concepts_output, concepts_real, batch_ids)
            self.concepts_last_step.extend(last_step_concept_accs)
            per_concept_last_step_concept_accs = type(self).calculate_step_acc(concepts_output, concepts_real, batch_ids, take_total_for_classes=False)
            self.per_concept_last_step.extend(per_concept_last_step_concept_accs)
            self.concepts_last_step_total += len(last_step_concept_accs)
        formula_last_step_accs = type(self).calculate_step_acc(type(self).get_outputs(formula_preds), real, batch_ids)
        self.formula_preds_last_step.extend(formula_last_step_accs)
        self.formula_preds_last_step_total += len(formula_last_step_accs)


    def prepare_constants(self, batch):
        SIZE = batch.num_nodes
        # we make at most |V|-1 steps
        GRAPH_SIZES = torch_scatter.scatter(torch.ones_like(batch.batch), batch.batch, reduce='sum')
        STEPS_SIZE = GRAPH_SIZES.max()
        return SIZE, STEPS_SIZE

    def set_initial_last_states(self, batch):
        self.last_latent = torch.zeros(batch.num_nodes, _DIM_LATENT, device=_DEVICE)
        self.last_continue_logits = torch.ones(batch.num_graphs, device=_DEVICE)
        self.last_concepts_logits = torch.zeros((batch.num_nodes, self.concept_features), device=_DEVICE)
        self.last_concepts_real = torch.zeros((batch.num_nodes, self.concept_features), device=_DEVICE).int()

        self.last_output_logits = torch.where(batch.x[0, :, 1].bool().unsqueeze(-1), 1e3, -1e3)
        self.last_output = (self.last_output_logits > 0).float()
        self.last_preds_with_formula = torch.zeros((batch.num_nodes, self.output_features), device=_DEVICE)

    def update_states(self, current_latent, concepts_logits, concepts_real,
                      output_logits, preds_with_formula, continue_logits, concept_continue_logits, concepts_continue_real=None):
        def update_per_mask(before, after, mask=None):
            # NOTE: this does expansion of the mask, if you do
            # NOT use expansion, use torch.where
            if mask is None:
                mask = self.mask
            mask = mask.unsqueeze(-1).expand_as(before)
            return torch.where(mask, after, before)
        self.last_continue_logits = torch.where(self.mask_cp, continue_logits,
                                                self.last_continue_logits)
        self.last_latent = update_per_mask(self.last_latent, current_latent)
        self.last_output_logits = update_per_mask(self.last_output_logits, output_logits)
        self.last_output = type(self).get_outputs(self.last_output_logits).float()

        if self.use_concepts:
            self.last_concepts_logits = update_per_mask(
                self.last_concepts_logits, concepts_logits)
        if self.use_concepts_sv:
            self.last_concepts_real = update_per_mask(self.last_concepts_real, concepts_real)
        if preds_with_formula is not None:
            self.last_preds_with_formula = update_per_mask(self.last_preds_with_formula, preds_with_formula)

    def prepare_initial_masks(self, batch):
        self.mask = torch.ones_like(batch.batch, dtype=torch.bool, device=_DEVICE)
        self.mask_cp = torch.ones(batch.num_graphs, dtype=torch.bool, device=_DEVICE)
        self.edge_mask = torch.ones_like(batch.edge_index[0], dtype=torch.bool, device=_DEVICE)


    def get_concept_loss(self,
                         mask,
                         concepts_real,
                         concepts_logits,
                         reduction='sum'):
        if not self.use_concepts_sv:
            return 0
        concepts_logits_masked = concepts_logits[mask]
        concepts_real_masked = concepts_real[mask]
        weight = torch.ones_like(concepts_logits_masked[0])

        if type(self) == models.AlgorithmColoring:
            weight[1] = 10
        loss_concepts = torch.nn.functional.binary_cross_entropy_with_logits(
            concepts_logits_masked,
            concepts_real_masked.float(),
            reduction=reduction,
            pos_weight=getattr(self.dataset, 'concept_pos_weights', None),
            weight=weight)

        return loss_concepts

    def get_losses_dict(self, validation=False):
        concepts_mul = 1*int(self.use_concepts_sv)
        # NOTE Here we do the averaging. The sum (not sum of mean-reduced losses!!!)
        # is averaged over the sum of steps (for termination outputs/logits) or the sum of
        # all nodes ever processed (for termination outputs/logits)

        # NOTE 2, note that for training, these losses are average per-batch, whereas
        # for validation, these losses are averaged over the whole val/testing set.
        if self.use_decision_tree:
            dtmul = 0
        else:
            dtmul = 1

        if self.hardcode_outputs:
            outmul = 0
        else:
            outmul = 1

        freeze_mul = 0 if not self.concepts_frozen() else 1

        if validation:
            losses_dict = {
                'total_loss_concepts': self.concepts_unlearned_mul*concepts_mul*self.losses['total_loss_concepts'] / (self.validation_sum_of_processed_nodes * self.concept_features),
                'total_loss_output': self.lambda_mul*freeze_mul*outmul*dtmul*self.losses['total_loss_output'] / (self.validation_sum_of_processed_nodes * self.output_features),
                'total_loss_term': self.lambda_mul*1*self.losses['total_loss_term'] / self.validation_sum_of_steps,
            }  if self.validation_sum_of_processed_nodes else 0
        else:
            losses_dict = {
                'total_loss_concepts': self.concepts_unlearned_mul*concepts_mul * self.losses['total_loss_concepts'] / (self.sum_of_processed_nodes * self.concept_features),
                'total_loss_output': self.lambda_mul*outmul*dtmul * self.losses['total_loss_output'] / (self.sum_of_processed_nodes * self.output_features),
                'total_loss_term': self.lambda_mul*1*self.losses['total_loss_term'] / self.sum_of_steps,
            } if self.sum_of_processed_nodes else 0

        if self.use_concepts and self.L1_loss:
            losses_dict.update(
                {'L1_loss': 0.005*dtmul*(l1_loss(self.concept_decoder))}# + l1_loss(self.termination_network))}
            )

        return losses_dict


    def get_training_loss(self):
        return sum(self.get_losses_dict().values()) if self.get_losses_dict() != 0 else 0

    def get_validation_losses(self):
        return sum(self.get_losses_dict(validation=True).values()) if self.get_losses_dict(validation=True) != 0 else 0

    def get_validation_accuracies(self):
        assert not torch.isnan(torch.tensor(self.mean_step)).any(), torch.tensor(self.mean_step)
        assert not torch.isnan(torch.tensor(self.concepts_mean_step)).any()
        assert self.last_step_total == len(self.last_step)
        assert self.concepts_last_step_total == len(self.concepts_last_step)
        assert self.formula_preds_last_step_total == len(self.formula_preds_last_step)
        if self.use_concepts and self.use_concepts_sv:
            self.pcls = torch.stack(self.per_concept_last_step, dim=0)
            self.pcms = torch.stack(self.per_concept_mean_step, dim=0)
        return {
            'mean_step_acc': torch.tensor(self.mean_step).sum()/len(self.mean_step),
            'concepts_mean_step_acc': torch.tensor(self.concepts_mean_step).sum()/len(self.concepts_mean_step) if len(self.concepts_mean_step) else 0,
            'per_concept_mean_step_acc': self.pcms.sum(0)/len(self.concepts_mean_step) if len(self.concepts_mean_step) and self.use_concepts_sv else 0,
            'formula_mean_step_acc': torch.tensor(self.formula_preds_mean_step).sum()/(len(self.formula_preds_mean_step) if self.formula_preds_mean_step else 1), # to avoid div by 0
            'term_formula_mean_step_acc': torch.tensor(self.term_formula_preds_mean_step).sum()/(len(self.term_formula_preds_mean_step) if self.term_formula_preds_mean_step else 1), # to avoid div by 0
            'term_mean_step_acc': torch.tensor(self.term_preds_mean_step).sum()/(len(self.term_preds_mean_step) if self.term_preds_mean_step else 1), # to avoid div by 0
            'last_step_acc': torch.tensor(self.last_step).mean(),
            'concepts_last_step_acc': torch.tensor(self.concepts_last_step).mean() if self.use_concepts_sv else 0,
            'per_concept_last_step_acc': self.pcls.mean(0) if self.use_concepts_sv else 0,
            'formula_last_step_acc': torch.tensor(self.formula_preds_last_step).mean() if self.formula_preds_last_step else 0
        }

    def zero_hidden(self, num_nodes):
        self.hidden = torch.zeros(num_nodes, self.latent_features).to(get_hyperparameters()['device'])

    def loop_condition(self, termination, STEPS_SIZE):
        return (((not self.training and (self.steps <= 0 or termination.any())) or
                 (self.training and (self.steps <= 0 or termination.any()))) and
                 self.steps < STEPS_SIZE)

    def apply_decision_tree_to_concepts(self, concepts_logits):
        outputs = torch.tensor(self.cto_decision_tree.predict((concepts_logits > 0).long().cpu())).to(_DEVICE).unsqueeze(-1)
        if self.output_features > 1:
            outputs = F.one_hot(torch.tensor(self.cto_decision_tree.predict((concepts_logits > 0).long().cpu())).to(_DEVICE), self.output_features)
        # mk = outputs != y_curr
        output_logits = torch.where(outputs.bool(), 1e3, -1e3)
        return outputs, output_logits

    def loop_body(self,
                  batch,
                  inp,
                  y_curr,
                  concepts_real,
                  true_termination,
                  compute_losses,
                  concepts_continue_real=None,
                  extract_formulas=False,
                  apply_formulas=False,
                  fit_decision_tree=False,
                  apply_decision_tree=False):

        current_latent, concepts_logits, output_logits, concept_continue_logits, continue_logits =\
            self(
                batch,
                inp,
                batch.edge_index
            )

        if apply_decision_tree:
            outputs, output_logits = self.apply_decision_tree_to_concepts(concepts_logits)

        preds_with_formula = None
        if apply_formulas:
            assert not self.training
            assert self.explanations is not None
            assert self.termination_idx_combination is not None
            assert self.termination_truth_assignment is not None
            assert self.use_concepts
            preds_with_formula = []
            if self.output_features == 1:
                classes = [0, 1]
            else:
                classes = range(self.output_features)
            for cls in classes:
                acc, preds = deep_logic.logic.test_explanation(
                    self.explanations[cls],
                    cls,
                    concepts_logits,
                    y_curr.long(),
                    give_local=True)
                if acc == 0:
                    preds = torch.zeros_like(preds)
                preds_with_formula.append(preds)

            preds_with_formula = torch.stack(preds_with_formula, dim=-1).float()
            if self.output_features == 1:
                preds_with_formula = preds_with_formula[:, 1].unsqueeze(-1)

            relevant_concepts = concept_continue_logits[:, self.termination_idx_combination] > 0
            relevant_concepts_eq = (relevant_concepts == self.termination_truth_assignment).all(dim=-1)
            termination_with_formula = torch_scatter.scatter(relevant_concepts_eq.float(), batch.batch, reduce='max')
        termination = termination_with_formula.float() if apply_formulas else continue_logits

        self.debug_batch = batch
        self.debug_y_curr = y_curr
        self.update_states(current_latent, concepts_logits, concepts_real,
                           output_logits, preds_with_formula, termination, concept_continue_logits, concepts_continue_real=concepts_continue_real)

        self.aggregate_loss_steps_and_acc(
            batch, self.mask, self.mask_cp,
            y_curr, output_logits, preds_with_formula,
            concepts_real, concepts_logits,
            concept_continue_logits,
            true_termination, continue_logits, termination,
            compute_losses=compute_losses)

        if extract_formulas or fit_decision_tree:
            assert self.use_concepts
            self.predictions['outputs'].append(self.last_output_logits.clone().detach())
            self.actual['outputs'].append(y_curr.clone().detach())
            self.predictions['concepts'].append(self.last_concepts_logits.clone().detach())
            self.actual['concepts'].append(self.last_concepts_real.clone().detach())
            self.predictions['terminations'].append(self.last_continue_logits.clone().detach())
            self.actual['terminations'].append(true_termination.clone().detach())

    def get_input_from_output(self, output, batch=None):
        if self.use_decision_tree and not self.apply_decision_tree:
            output = batch.x[min(self.steps+1, len(batch.x)-1), :, 1]
        output = (output.long() > 0)
        return F.one_hot(output.long().squeeze(-1), num_classes=self.node_features).float()

    def get_step_output(self, batch, step):
        output_logits = torch.where(batch.y[step, :, 1].bool().unsqueeze(-1), 1e3, -1e3)
        output = (output_logits > 0).float()
        return output_logits, output

    def get_step_input(self, x_curr, batch):
        return x_curr if (self.training and self.use_TF) or (self.use_decision_tree and not self.apply_decision_tree) else self.get_input_from_output(self.last_output_logits, batch)

    def concepts_frozen(self):
        return True
        return list(self.concept_decoder.parameters())[0].requires_grad

    def toggle_freeze_concepts(self):
        utils.toggle_freeze_module(self.decoder_network)
        utils.toggle_freeze_module(self.decoder_network_term)
        print("TOGGLED", list(self.decoder_network.parameters())[0].requires_grad)

    def process(
            self,
            batch,
            EPSILON=0,
            enforced_mask=None,
            extract_formulas=False,
            apply_formulas=False,
            fit_decision_tree=False,
            apply_decision_tree=False,
            compute_losses=True,
            hardcode_concepts=None,
            hardcode_outputs=False,
            debug=False):
        '''
        Method that takes a batch, does all iterations of every graph inside it
        and accumulates all metrics/losses.
        '''

        SIZE, STEPS_SIZE = self.prepare_constants(batch)
        self.hardcode_concepts = hardcode_concepts
        self.hardcode_outputs = hardcode_outputs
        self.apply_decision_tree = apply_decision_tree

        # Pytorch Geometric batches along the node dimension, but we execute
        # along the temporal (step) dimension, hence we need to transpose
        # a few tensors. Done by `prepare_batch`.
        batch = utils.prepare_batch(batch)
        if self.drop_last_concept:
            batch.concepts = batch.concepts[:, :, :-1]
            batch.last_concepts_real = batch.last_concepts_real[:, :-1]
        # When we want to calculate last step metrics/accuracies
        # we need to take into account again different termination per graph
        # hence we save last step tensors (e.g. outputs/concepts) into their
        # corresponding tensor. The function below prepares these tensors
        # (all set to zeros, except masking for computation, which are ones)
        self.set_initial_last_states(batch)
        # Prepare masking tensors (each graph does at least 1 iteration of the algo)
        self.prepare_initial_masks(batch)
        # A flag if we had a wrong graph in the batch. Used for visualisation
        # of what went wrong
        self.wrong_flag = False
        assert self.mask_cp.all(), self.mask_cp
        to_process = torch.ones([batch.num_graphs], device=_DEVICE)

        while self.loop_condition(
                batch.termination[self.steps-1] if self.training or fit_decision_tree or not self.concepts_frozen() else self.mask_cp,
                STEPS_SIZE):

            # take inputs/target values/concepts
            x_curr, y_curr = batch.x[self.steps], batch.y[self.steps]
            concepts_real = batch.concepts[self.steps]
            concepts_continue_real = batch.concepts[min(self.steps+1, len(batch.concepts)-1)]
            if not self.training:
                assert (self.last_continue_logits > 0).any() or True

            # Some algorithms, e.g. parallel colouring outputs fewer values than it takes
            # (e.g. priorities for colouring are unchanged one every step)
            # so if we reuse our last step outputs, they need to be fed back in.
            # NOTE self.get_step_input always takes x_curr, if we train
            inp = self.get_step_input(x_curr, batch)

            true_termination = batch.termination[self.steps] if self.steps < STEPS_SIZE else torch.zeros_like(batch.termination[self.steps-1])

            # Does one iteration of the algo and accumulates statistics
            self.loop_body(batch,
                           inp,
                           y_curr,
                           concepts_real,
                           true_termination,
                           compute_losses,
                           extract_formulas=extract_formulas,
                           apply_formulas=apply_formulas,
                           fit_decision_tree=fit_decision_tree,
                           apply_decision_tree=apply_decision_tree,
                           concepts_continue_real=concepts_continue_real)
            # And calculate what graphs would execute on the next step.
            self.mask, self.mask_cp = type(self).get_masks(self.training, batch, true_termination if  fit_decision_tree or not self.concepts_frozen() else self.last_continue_logits, enforced_mask)
        if not self.training:
            # After we are done with the execution, accumulate statistics related
            # to last step accuracies.

            self.aggregate_last_step(
                batch.batch,
                self.last_output_logits,
                batch.y[-1],
                self.last_concepts_logits > 0,
                batch.last_concepts_real, self.last_preds_with_formula)

    @staticmethod
    def get_outputs(outputs):
        return outputs > 0

    def compute_concepts_and_outputs(self, batch, encoded_nodes, hidden, next_step=False):


        if self.use_concepts:
            concepts_logits = self.decoder_network(torch.cat((encoded_nodes, hidden), dim=1))

            real_concepts_logits = torch.where(batch.concepts[min(self.steps+int(next_step), len(batch.concepts)-1)].bool(), 1e3, -1e3)
            if self.hardcode_concepts is not None:
                concepts_logits[:, self.hardcode_concepts] = real_concepts_logits[:, self.hardcode_concepts]


            if not self.use_decision_tree:
                output_logits = self.concept_decoder(concepts_logits)
            elif hasattr(self, 'cto_decision_tree'):
                outputs = torch.tensor(self.cto_decision_tree.predict((concepts_logits > 0).long().cpu())).to(_DEVICE)
                if self.output_features > 1:
                    outputs = F.one_hot(torch.tensor(self.cto_decision_tree.predict((concepts_logits > 0).long().cpu())).to(_DEVICE), self.output_features)
                output_logits = torch.where(outputs.bool(), 1e3, -1e3)
            else:
                output_logits = self.concept_decoder(concepts_logits)


        else:
            concepts_logits = None
            output_logits = self.decoder_network(torch.cat((encoded_nodes, hidden), dim=1))

        if self.hardcode_outputs or not self.concepts_frozen():
            
            if type(self) == models.AlgorithmColoring:
                output_logits = torch.where(F.one_hot(batch.y[self.steps], num_classes=self.output_features).bool(), 1e3, -1e3).float()
            else:
                output_logits = torch.where(batch.y[self.steps].bool(), 1e3, -1e3).unsqueeze(-1)
        return concepts_logits, output_logits

    def encode_nodes(self, current_input, last_latent):
        return self.node_encoder(torch.cat((current_input, last_latent), dim=1))

    def forward(self, batch, current_input, edge_index):
        batch_ids = batch.batch

        assert not torch.isnan(self.last_latent).any()
        assert not torch.isnan(current_input).any()
        encoded_nodes = self.encode_nodes(current_input, self.last_latent)
        hidden = self.processor(encoded_nodes, edge_index, self.last_latent)
        assert not torch.isnan(hidden).any()
        assert not torch.isnan(encoded_nodes).any()
        concepts_logits, output_logits = self.compute_concepts_and_outputs(batch, encoded_nodes, hidden)
        assert not torch.isnan(output_logits).any(), hidden[torch.isnan(output_logits).squeeze()]
        if self.next_step_pool:
            if self.apply_decision_tree:
                output_logits, _ = self.apply_decision_tree_to_concepts(concepts_logits)
            inp2 = self.get_input_from_output(output_logits, batch=batch)

            encoded_nodes2 = self.encode_nodes(inp2, hidden)
            hidden2 = self.processor(encoded_nodes2, edge_index, hidden)
            concepts_logits2, _ = self.compute_concepts_and_outputs(batch, encoded_nodes2, hidden2, next_step=True)
            catted = hidden2

        if not self.next_step_pool:
            catted = hidden

        concept_continue_logits, continue_logits = self.get_continue_logits(
            batch_ids,
            catted,
            sth_else=torch.cat(
                (concepts_logits > 0, ), dim=-1)
            if concepts_logits is not None else None)
        return hidden, concepts_logits, output_logits, concepts_logits2 if self.next_step_pool else concepts_logits, continue_logits
