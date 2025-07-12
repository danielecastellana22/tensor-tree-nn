import matplotlib
import matplotlib.pyplot as plt
from networkx.drawing.nx_agraph import graphviz_layout
import networkx as nx


def plot_tree_sentence(dgl_G, idx2words, input_attr='x', output_attr='y', type_attr='t', idx2types=None, ax=None, vmin=0, vmax=4):
    node_attr_list = [input_attr]

    if output_attr is not None:
        node_attr_list.append(output_attr)
    if type_attr is not None:
        node_attr_list.append(type_attr)

    G = dgl_G.to_networkx(node_attr_list)
    pos = graphviz_layout(G, prog='dot')
    # invert the y-axis
    for n in pos:
        pos[n] = (pos[n][0], -pos[n][1])

    cmap = matplotlib.cm.get_cmap('bwr')
    my_color_map = {-1: (1, 200/255, 0, 1)}
    for v in range(vmin, vmax+1):
        my_color_map[v] = cmap((v-vmin)/(vmax-vmin))
    #my_color_map[2] = (0.9, 0.9, 0.9)
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
        elif type_id != -1 and idx2types is not None:
                lbls[u] = idx2types[type_id]
        else:
            lbls[u] = u

    if output_attr is not None:
        colors = [my_color_map[G.nodes[u][output_attr].item()] for u in G.nodes]
    else:
        colors = [my_color_map[-1] for u in G.nodes]

    nx.draw(G, pos=pos, with_labels=True, labels=lbls, node_color=colors, ax=ax)


def plot_netwrokx_tree(nx_t, node_attr_list=None, edge_attr=None, ax=None, **kwargs):
    pos = graphviz_layout(nx_t, prog='dot', args='-Granksep=2 -Gnodesep=2')
    if ax is None:
        plt.figure(figsize=(15, 15))
        ax = plt.gca()
    # invert the y-axis
    for n in pos:
        pos[n] = (pos[n][0], -pos[n][1])

    #node_color = ['#1f78b4' for i in range(nx_t.number_of_nodes())]

    node_labels = {}
    if node_attr_list is not None:
        for id, x in nx_t.nodes(data=True):
            aux = [str(id)]
            for att in node_attr_list:
                if att in x:
                    aux.append(str(x[att]))
            node_labels[id] = '|'.join(aux)
    else:
        node_labels = {u:u for u in nx_t.nodes}
    nx.draw_networkx(nx_t, pos=pos, labels=node_labels, ax=ax, **kwargs)
    if edge_attr is not None:
        edge_labels = {(u, v): d[edge_attr] for u, v, d in nx_t.edges(data=True) if edge_attr in d}
        nx.draw_networkx_edge_labels(nx_t, pos=pos, edge_labels=edge_labels, ax=ax)


def tree_sentence_to_tikz(dgl_G, idx2words, node_attr='x'):
    rev_t = dgl_G.to_networkx(node_attr).reverse()

    def rec_writing(u, ind):
        out = ''
        l = rev_t.nodes[u][node_attr].item()
        if l == -1:
            l = u
        else:
            l = idx2words[l]


        attr = '[]' if rev_t.out_degrees(u) == 0 else '[internal]'
        if ind == 0:
            out += '\\node{} {{{}}}'.format(attr, l)
        else:
            out += '\t' * ind + 'node{} {{{}}}'.format(attr, l)
        out += '\n'

        for ch in rev_t.successors(u):
            out += '\t' * ind + 'child{\n'
            out += rec_writing(ch, ind + 1)
            out += '\t' * ind + '}\n'
        if ind == 0:
            out += ';'
        return out

    return rec_writing(0, 0)

