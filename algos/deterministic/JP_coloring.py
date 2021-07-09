import torch
import torch_geometric
from torch_geometric.data import Data
import torch_scatter
import seaborn as sns

import networkx as nx
import matplotlib
import matplotlib.pyplot as plt

from algos.layers.encoders import integer2bit
from algos.hyperparameters import get_hyperparameters

_POS = None

def _draw_pt_graph(G, priority, colors=None):
    G = torch_geometric.utils.to_networkx(G)
    palette = [sns.color_palette("Paired").as_hex()[i] for i in [0, 1, 2, 3, 4, 5]]
    cmap = matplotlib.colors.ListedColormap(palette)
    global _POS
    pos = nx.spring_layout(G, k=.15, iterations=10) if _POS is None else _POS
    if _POS is None:
        _POS = dict(pos.items())
    num_nodes = len(G.nodes)
    plt.figure(figsize=(15, 15))
    nx.draw_networkx_nodes(G,
            pos,
            node_list=G.nodes(),
            node_color=colors,
            cmap=cmap,
            node_size=1200,
            alpha=0.9,
            with_labels=True,
    )
    nx.draw_networkx_edges(G, pos)
    nx.draw_networkx_labels(G, pos,
        dict(zip(range(num_nodes), zip(range(num_nodes), priority.tolist()))), font_size=8)
    plt.axis('off')
    plt.draw()

def jones_plassmann(G, num_colors=5):
    '''

    The below algorithm is based on:
    M.T.Jones and, P.E.Plassmann, A Parallel Graph Coloring Heuristic, SIAM, Journal of Scienti c Computing 14 (1993) 654

    The algorithm takes a graph (a networkx class), randomly assigns a priority
    to each node and colours according to the above paper. In a nutshell:
        - assume there is a colour order, (e.g. color 1 < color 2 < ... < color |C|,
          where |C|=number of colours)
        
        - every node checks if it is uncoloured and has the highest priority

        - if it has these two properties then it colours itself in the lowest
          colour (according to the ordering) not seen in the neighbourhood.

        - the only differences to J.P. algorithm is that we assume a fixed
          number of colours (e.g. 5) as NN are known not to be able to 'count'
          (ref needed) and we resample the priorities whenever there is a clash
          (two nodes with the same priority) or we need more colours than needed
    '''

    # we have |C|+1 output classes, 1 for uncolored node and |C|=5 for each colour
    # Concept encoding
    # f0 = colored or not
    # f1 = largest uncolored in neighb. or not (i.e. has highest priority in neighb.)
    # f2-f6 = colors seen around 1 to 5
    # f7-f11 = has been colored in 1 to 5
    # E.g. for having colour 3 on the next iteration an ideal explanation is:
    # (f0 & f9) | (~f0 & f1 & f2 & f3 & ~f4)
    num_nodes = len(G.nodes)
    priority = list({'x': x} for x in torch.randint(0, 255, (num_nodes,)).tolist())
    priority = dict(zip(range(num_nodes), priority))
    nx.set_node_attributes(G, priority)
    G = torch_geometric.utils.from_networkx(G)
    colored = torch.zeros(num_nodes, dtype=torch.long)
    all_inp, all_target_col, all_target_concepts, all_term = [], [], [], []
    all_target_concepts_fin = []
    n1, n2 = G.edge_index.tolist()
    inp_priority = integer2bit(G.x)
    last_concepts_real = None
    for _ in range(num_nodes+1):
        c1h = torch.nn.functional.one_hot(colored, num_classes=num_colors+1)
        all_inp.append(torch.cat((c1h, inp_priority), dim=-1).clone())
        concepts = torch.zeros((num_nodes, get_hyperparameters()['dim_concept_parallel_coloring']), dtype=torch.int)
        concepts[colored != 0, 0] = 1
        priority = G.x.clone()
        priority[colored != 0] = -1
        edge_priority = priority[n2]
        received_priority = torch.full((num_nodes,), -1)
        torch_scatter.scatter(edge_priority.long(), torch.tensor(n1, dtype=torch.long), reduce='max', out=received_priority)

        if ((received_priority == priority) & (priority != -1)).any():
            print("REDO: clashing priorities")
            return None

        to_be_colored = (colored == 0) & (received_priority < priority)
        concepts[to_be_colored, 1] = 1
        colors_around = torch.zeros(num_nodes, num_colors, dtype=torch.bool)
        for i in range(num_colors):
            colors_sent = colored[n2] == i+1
            rec_color_i = torch.full((num_nodes,), -1)
            torch_scatter.scatter(colors_sent.long(), torch.tensor(n1, dtype=torch.long), reduce='max', out=rec_color_i)
            colors_around[rec_color_i != -1, i] = rec_color_i[rec_color_i != -1].bool()


        colors_to_receive = colors_around.int().min(dim=-1).indices+1
        if colors_around.all(dim=-1).any():
            print("REDO: colors not enough")
            return None
        colored = torch.where(to_be_colored, colors_to_receive, colored)
        concepts[:, 2:7] = colors_around
        concepts_fin = concepts.min(dim=0).values.unsqueeze(0)
        all_target_concepts.append(concepts.clone())
        all_target_concepts_fin.append(concepts_fin.clone())
        all_target_col.append(colored.clone())
        all_term.append((colored == 0).any().unsqueeze(-1))
        if not (colored == 0).any() and last_concepts_real is None:
            last_concepts_real = concepts.clone()

    all_target_concepts_fin = [all_target_concepts_fin[i + 1]
                               for i in range(0, len(all_target_concepts_fin)-1)] + [all_target_concepts_fin[-1]]
    data = Data(torch.stack(all_inp, dim=1),
                edge_index=torch.tensor(G.edge_index),
                y=torch.stack(all_target_col, dim=1),
                concepts=torch.stack(all_target_concepts, dim=1),
                last_concepts_real=last_concepts_real,
                concepts_fin=torch.stack(all_target_concepts_fin, dim=1),
                priorities=inp_priority,
                termination=torch.stack(all_term, dim=1))
    return data
