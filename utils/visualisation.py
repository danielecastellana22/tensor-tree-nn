import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from networkx.drawing.nx_agraph import graphviz_layout
import networkx as nx
import os
import pickle
import torch as th


def __plot_matrix__(ax, cm, x_label, x_tick_label, y_label, y_tick_label, title=None, cmap='viridis', vmin=None, vmax=None, fmt='.2f'):
    if vmin is None:
        vmin = cm.min()

    if vmax is None:
        vmax = cm.max()

    cmap_obj = matplotlib.cm.get_cmap(cmap)

    im = ax.imshow(cm, interpolation='nearest', cmap=cmap_obj, vmin=vmin, vmax=vmax)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=x_tick_label, yticklabels=y_tick_label,
           title=title,
           ylabel=y_label,
           xlabel=x_label)

    ax.set_xticks(np.arange(cm.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(cm.shape[0]+1)-.5, minor=True)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.

    def get_text_color(v):
        v = (v-vmin)/(vmax-vmin)
        rgba = cmap_obj(v)
        lum=0.2126*rgba[0] + 0.7152*rgba[1] + 0.0722*rgba[2]
        if lum<0.5:
            return 'white'
        else:
            return 'black'

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color=get_text_color(cm[i, j]))

    return ax


def plot_results(results, param_grid, params_to_plot, params_to_change, metrics_to_plot, params_to_maximise=None, figsize=(30,10)):
    if len(params_to_plot) > 2:
        raise ValueError('Too many parameters to plot.')

    if len(params_to_change) > 1:
        raise ValueError('Too many parameters to plot.')

    param_keys = list(param_grid.keys())
    idx_params_to_plot = [param_keys.index(pp) for pp in params_to_plot]
    idx_params_to_change = [param_keys.index(pp) for pp in params_to_change]

    if params_to_maximise is None:
        # all the other indexes.
        mask = [True] * len(param_keys)
        for i in idx_params_to_plot:
            mask[i] = False

        for i in idx_params_to_change:
            mask[i] = False

        # the last axis is for the average
        mask[-1] = False

        params_to_maximise = np.array(param_keys)[mask]

    idx_params_to_max = [param_keys.index(pp) for pp in params_to_maximise]

    # mean over last dimension and max over idx_params_to_max
    n_params = len(param_keys)
    metric_results = []
    for k in metrics_to_plot:
        metric_results.append(
            results[k].mean(n_params - 1, keepdims=True).max(axis=tuple(idx_params_to_max), keepdims=True))

    # -1 because we reemove the run
    idx_list = [0] * (n_params)
    for i, idx in enumerate(idx_params_to_plot):
        param_name = params_to_plot[i]
        idx_list[idx] = slice(0, len(param_grid[param_name]))

    n_row = len(metrics_to_plot)
    n_col = len(param_grid[params_to_change[0]])
    fig, ax_list = plt.subplots(n_row, n_col, figsize=figsize)
    ax_list = np.array(ax_list).reshape(n_row, n_col)
    for i, v in enumerate(param_grid[params_to_change[0]]):
        idx_list[idx_params_to_change[0]] = i
        for j, m in enumerate(metrics_to_plot):
            __plot_matrix__(ax_list[j, i],
                            100 * metric_results[j][tuple(idx_list)],
                            y_label=params_to_plot[0],
                            y_tick_label=param_grid[params_to_plot[0]],
                            x_label=params_to_plot[1],
                            x_tick_label=param_grid[params_to_plot[1]],
                            title='{} and {}={}'.format(m, params_to_change[0], v))

    fig.tight_layout()


def plot_confusion_matrix(ax, cm, classes_name, title=None):
    #cm = cm / cm.sum()
    __plot_matrix__(ax, cm,
                    x_label='Predicted label', x_tick_label=classes_name, fmt='d',
                    y_label='True label', y_tick_label=classes_name, title=title, cmap='Blues')


def print_best_configuration(all_results, param_grid, metrics_list):
    for k in metrics_list:
        print('{0} {1} {0}'.format('-' * 70, k.upper()))
        run_axis = list(param_grid.keys()).index('run')
        mean_results = all_results[k].mean(axis=run_axis)
        std_results = all_results[k].std(axis=run_axis)

        ravel_idx_best_conf = np.argmax(mean_results)
        idx_best_conf = np.unravel_index(ravel_idx_best_conf, mean_results.shape)

        print('Best node accuracy: {:0.2f} \xB1 {:0.2f}'.format(100 * mean_results[idx_best_conf],
                                                                100 * std_results[idx_best_conf]))

        s = 'Best configuration {}: \t'.format(ravel_idx_best_conf)
        for i, k in enumerate(param_grid):
            if i != run_axis:
                s += '{}: {}\t'.format(k, param_grid[k][idx_best_conf[i]])
        print(s)


def plot_tree_sentence(dgl_G, idx2words, input_attr='x', output_attr='y', type_attr='type_id', idx2types=None, ax=None):
    G = dgl_G.to_networkx([input_attr, output_attr, type_attr])
    pos = graphviz_layout(G, prog='dot')
    # invert the y-axis
    for n in pos:
        pos[n] = (pos[n][0], -pos[n][1])

    vmin = 0
    vmax = 4
    cmap = matplotlib.cm.get_cmap('bwr')
    my_color_map = {-1: (1, 200/255, 0, 1)}
    for v in range(vmin, vmax+1):
        my_color_map[v] = cmap((v-vmin)/(vmax-vmin))
    my_color_map[2] = (0.9, 0.9, 0.9)
    #print(my_color_map)

    lbls = {}
    for u in G.nodes:
        x = G.nodes[u][input_attr].item()
        if type_attr in G.nodes[u]:
            type_id = G.nodes[u][type_attr].item()
        else:
            type_id = -1
        if x != -1:
            lbls[u] = idx2words[x]
        elif type_id != -1:
            if idx2types is not None:
                lbls[u] = idx2types[type_id]

    colors = [my_color_map[G.nodes[u][output_attr].item()] for u in G.nodes]

    nx.draw(G, pos=pos, with_labels=True, labels=lbls, node_color=colors, ax=ax)


def plot_netwrokx_tree(nx_t, node_attr=None, edge_attr=None, ax=None):
    pos = graphviz_layout(nx_t, prog='dot')

    # invert the y-axis
    for n in pos:
        pos[n] = (pos[n][0], -pos[n][1])

    node_color = ['#1f78b4' for i in range(nx_t.number_of_nodes())]

    node_labels = {}
    if node_attr is not None:
        for id, x in nx_t.nodes(data=True):
            aux= []
            for att in node_attr:
                if att in x:
                    aux.append(str(x[att]))
            node_labels[id] = '|'.join(aux)
    nx.draw_networkx(nx_t, pos=pos, labels=node_labels, node_color=node_color, ax=ax)
    if edge_attr is not None:
        edge_labels = {(u, v): d[edge_attr] for u, v, d in nx_t.edges(data=True) if edge_attr in d}
        nx.draw_networkx_edge_labels(nx_t, pos=pos, edge_labels=edge_labels, ax=ax)

def read_checkpoint(folder_name, verbose=0):

    with open(os.path.join(folder_name, 'checkpoint.pickle'), mode='rb') as f:
        ms_checkpoint = pickle.load(f)

    param_keys = list(ms_checkpoint['param_list'][0].keys())
    param_grid = {}
    # get configuration shape
    new_shape = []
    for k in param_keys:
        s = set([x[k] for x in ms_checkpoint['param_list']])
        new_shape.append(len(s))
        param_grid[k] = sorted(list(s))
    n_run = new_shape[-1]
    n_conf = np.prod(new_shape)
    if verbose:
        print('The grid paramters are: {}'.format(param_keys))
        print('The grid is :')
        print(param_grid)
    all_results = {}
    metric_keys = list(ms_checkpoint['results'][0].keys())
    for k in metric_keys:
        all_results[k] = [x[k] for x in ms_checkpoint['results']] + [0 for i in range(np.prod(new_shape) - len(ms_checkpoint['results']))]
        all_results[k] = np.reshape(all_results[k], new_shape)
    if verbose:
        print('The metrics are: {}'.format(metric_keys))
        print('Each item contains {} results.'.format(all_results[metric_keys[0]].size))
    return all_results, param_grid, param_keys, metric_keys


def get_best_models(ms_folder, idx_best_conf):
    best_m = []
    for i in idx_best_conf:
        path = os.path.join(ms_folder, 'k_{}'.format(i))
        path = os.path.join(path,'best_model.pkl')
        with open(path, mode='rb') as f:
            best_m.append(th.load(f, map_location='cpu'))
    return best_m