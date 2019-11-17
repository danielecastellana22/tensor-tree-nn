import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from networkx.drawing.nx_agraph import graphviz_layout
import networkx as nx
#import dgl


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


def plot_tree_sentence(dgl_G, idx2word):
    G = dgl_G.to_networkx(['x', 'y'])
    pos = graphviz_layout(G, prog='dot')
    # invert the y-axis
    for n in pos:
        pos[n] = (pos[n][0], -pos[n][1])

    # plt.figure(figsize=(9,3))

    cmap_name = 'coolwarm'
    vmin = 0
    vmax = 4

    plt.figure(figsize=(13, 7))
    lbls = {u: idx2word[G.nodes[u]['x'].numpy()] for u in G.nodes}
    colors = [G.nodes[u]['y'].numpy() for u in G.nodes]
    nx.draw(G, pos=pos, with_labels=True, labels=lbls, node_color=colors, cmap=cmap_name, vmin=vmin, vmax=vmax)
    plt.axis('off')

    plt.show()


def plot_tree(dgl_G, attr_to_print):
    G = dgl_G.to_networkx(attr_to_print)
    pos = graphviz_layout(G, prog='dot')
    # invert the y-axis
    for n in pos:
        pos[n] = (pos[n][0], -pos[n][1])

    # plt.figure(figsize=(9,3))

    cmap_name = 'coolwarm'
    vmin = 0
    vmax = 4

    fig, ax_list = plt.subplots(1, len(attr_to_print), figsize=(30, 10))

    for i, k in enumerate(attr_to_print):
        plt.sca(ax_list[i])
        lbls = {u: G.nodes[u][k].numpy() for u in G.nodes}
        colors = [G.nodes[u][k].numpy() for u in G.nodes]
        nx.draw(G, pos=pos, with_labels=True, labels=lbls, node_color=colors, cmap=cmap_name, vmin=vmin, vmax=vmax)
        plt.axis('off')
        plt.title(k)

    plt.show()