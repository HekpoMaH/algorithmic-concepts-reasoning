"""
Usage:
    plot_per_epoch.py (--model-names=MN)... [--algos=ALGO]... [options] 

Options:
    -h --help           Show this screen.

    --algos ALGO        Which algorithms to load {BFS, parallel_coloring}.
                        Repeatable parameter. [default: BFS]

    --has-GRU           Does the processor utilise GRU unit? [default: False]

    --model-names MN    Names of models to load
    
    --num-nodes NN      Number of nodes in the graphs to test on [default: 20]

    --use-seeds         Use seeds for STD. It will automatically modify name as
                        appropriate. [default: False]

    --num-seeds NS      How many different seeds to plot. [default: 5]

"""
import os
import pandas
import schema
from docopt import docopt
import matplotlib.pyplot as plt
import seaborn as sns
import subprocess

def get_concept_labels(algoname):
    if algoname == 'BFS':
        return {'C1': 'hasBeenVisited', 'C2': 'hasVisitedNeighbour'}
    if algoname == 'parallel_coloring':
        return {
            'C1': 'isColored', 'C2': 'hasPriority',
            'C3': 'c1S', 'C4': 'c2S',
            'C5': 'c3S', 'C6': 'c4S',
            'C7': 'c5S',
        }
    return {}

def line_count(filename):
    return int(subprocess.check_output('wc -l {}'.format(filename), shell=True).split()[0])

def fill_between(axis, x, mean, std, clip_low=0, clip_high=None):
    axis.fill_between(
        x,
        (mean - std).clip(clip_low, clip_high),
        (mean + std).clip(clip_low, clip_high),
        alpha=0.1)

def widen_legend(axis, lw):
    for line in axis.get_legend().get_lines():
        line.set_linewidth(lw)

def plt_set(plt, lw):
    plt.xticks(fontsize=FONT_SZ)
    plt.yticks(fontsize=FONT_SZ)
    plt.ylim(bottom=0.95, top=1)
    plt.rcParams.update({'lines.markeredgewidth': lw})
    plt.rcParams.update({'lines.linewidth': lw})
    plt.rcParams.update({'errorbar.capsize': lw+3})

args = docopt(__doc__)
schema = schema.Schema({'--algos': schema.And(list, [lambda n: n in ['BFS', 'parallel_coloring']]),
                        '--help': bool,
                        '--has-GRU': bool,
                        '--use-seeds': bool,
                        '--num-seeds': schema.Use(int),
                        '--num-nodes': schema.Use(int),
                        '--model-names': schema.Or(None, schema.Use(list))})
args = schema.validate(args)
print(args)

if not os.path.exists('./algos/figures/'):
    os.makedirs('./algos/figures/')

if args['--use-seeds']:
    dfs = []
    for seed in range(args['--num-seeds']):
        results_path = './algos/results/'+args['--model-names'][0]+f'_seed_{seed}'+'.csv'
        dfs.append(pandas.read_csv(results_path, index_col=False))
    combined_results = pandas.concat(dfs)
    epoch_index_mean = combined_results.groupby(combined_results.epoch).mean().add_suffix('_mean').reset_index()
    epoch_index_std = combined_results.groupby(combined_results.epoch).std().add_suffix('_std').reset_index()
    results = pandas.concat((epoch_index_mean,epoch_index_std), axis='columns')
    results = results.loc[:,~results.columns.duplicated()]
else:
    results_path = './algos/results/'+args['--model-names'][0]+'.csv'
    results = pandas.read_csv(results_path)

results = results.iloc[::5, :]
print('results', results)
suffix = '_mean' if args['--use-seeds'] else ''

sns.set()
FONT_SZ = 88
LABELPAD = 20
LINEWIDTH = 3
LEGEND_LINEWIDTH = 6
plt.rcParams["figure.figsize"] = [42, 36]
fig, ax1 = plt.subplots()
plt_set(plt, LINEWIDTH)

ax1.set_xlabel('Epoch', fontsize=FONT_SZ)
ax1.set_ylabel('Loss', fontsize=FONT_SZ)
lines = []
for algo in args['--algos']:
    line = ax1.plot(results['epoch'], results[f'{algo}_loss{suffix}'], label=f'{algo} total loss')
    lines.extend(line)
    if args['--use-seeds']:
        fill_between(ax1, results['epoch'], results[f'{algo}_loss{suffix}'], results[f'{algo}_loss_std'])

ax2 = ax1.twinx()
ax2.set_ylabel('Accuracy', fontsize=FONT_SZ)
ax2.tick_params(axis='y', labelsize=FONT_SZ)
for algo in args['--algos']:
    line = ax2.plot(results['epoch'], results[f'{algo}_last_step_acc{suffix}'], linestyle='--', label=f'{algo} last step accuracy')
    lines.extend(line)
    if args['--use-seeds']:
        fill_between(ax2, results['epoch'], results[f'{algo}_last_step_acc{suffix}'], results[f'{algo}_last_step_acc_std'], clip_high=1)

    if results[f'{algo}_formula_last_step_acc{suffix}'][0] != -1:
        line = ax2.plot(results['epoch'], results[f'{algo}_formula_last_step_acc{suffix}'], linestyle='-.', label=f'{algo} formula last step accuracy')
        lines.extend(line)
        if args['--use-seeds']:
            fill_between(ax2, results['epoch'], results[f'{algo}_formula_last_step_acc{suffix}'], results[f'{algo}_last_step_acc_std'], clip_high=1)
labels = [l.get_label() for l in lines]

ax2.legend(lines, labels, ncol=3, loc='upper center', bbox_to_anchor=(0.48, 1.18), fontsize=FONT_SZ)
widen_legend(ax2, LEGEND_LINEWIDTH)

plt.savefig('./algos/figures/'+args['--model-names'][0]+'.png')

custom_palette = sns.color_palette([
    "#48bf8e", "#e54886", "#37b51f", "#801967", "#c0e15c", "#7a2edc",
    "darkorange", "#fe74fe", "#155126", "#f5b3da", "#673d17", "#aedddd"
])
sns.set(palette=custom_palette)
fig_per_concept, ax_per_concept = plt.subplots()
ax_per_concept.set_xlabel('Epoch', fontsize=FONT_SZ)
ax_per_concept.set_ylabel('Mean Step Accuracy', fontsize=FONT_SZ)
plt_set(plt, LINEWIDTH)

fig_per_concept_l, ax_per_concept_l = plt.subplots()
ax_per_concept_l.set_xlabel('Epoch', fontsize=FONT_SZ)
ax_per_concept_l.set_ylabel('Last Step Accuracy', fontsize=FONT_SZ)
plt_set(plt, LINEWIDTH-0.5)

lines = []
lines_l = []
for algo in args['--algos']:
    labels_concepts = get_concept_labels(algo)
    print("LABELS", labels_concepts)
    num_concepts = len(results.filter(regex=(f'{algo}_concept_[0-9]+_mean')).columns)
    if args['--use-seeds']:
        num_concepts //= 2
    print("ALGO", algo)
    print(num_concepts)
    rnc = list(range(num_concepts))
    if algo == 'parallel_coloring':
        rnc[1], rnc[-1] = rnc[-1], rnc[1]
    for i in rnc:
        line = ax_per_concept.plot(
            results['epoch'],
            results[f'{algo}_concept_{i}_mean_step_acc{suffix}'],
            label=labels_concepts.get(f'C{i+1}', f'$C_{{{i+1}}}$'))
        lines.extend(line)
        line = ax_per_concept_l.plot(
            results['epoch'],
            results[f'{algo}_concept_{i}_last_step_acc{suffix}'],
            label=labels_concepts.get(f'C{i+1}', f'$C_{{{i+1}}}$'))
        lines_l.extend(line)
        if args['--use-seeds']:
            fill_between(ax_per_concept, results['epoch'], results[f'{algo}_concept_{i}_mean_step_acc{suffix}'], results[f'{algo}_concept_{i}_mean_step_acc_std'], clip_high=1)
            fill_between(ax_per_concept_l, results['epoch'], results[f'{algo}_concept_{i}_last_step_acc{suffix}'], results[f'{algo}_concept_{i}_last_step_acc_std'], clip_high=1)

labels = [l.get_label() for l in lines]
labels_l = [l.get_label() for l in lines_l]
ax_per_concept.legend(lines, labels, ncol=4, loc='upper center', bbox_to_anchor=(0.48, 1.18), fontsize=FONT_SZ)
widen_legend(ax_per_concept, LEGEND_LINEWIDTH)
ax_per_concept_l.legend(lines_l, labels_l, ncol=4, loc='upper center', bbox_to_anchor=(0.48, 1.18), fontsize=FONT_SZ)
widen_legend(ax_per_concept_l, LEGEND_LINEWIDTH)
fig_per_concept.savefig('./algos/figures/'+args['--model-names'][0]+'_per_concept_mean_acc.png')
fig_per_concept_l.savefig('./algos/figures/'+args['--model-names'][0]+'_per_concept_last_acc.png')
