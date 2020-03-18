from nltk import Tree
import re
import dgl


def string_to_nltk_tree(s):
    node_pattern = '[^%s\t\n\r\f\v%s%s]+' % (re.escape(' '), re.escape('('), re.escape(')'))
    return Tree.fromstring(s, brackets='()', node_pattern=node_pattern, leaf_pattern=node_pattern)


def nx_to_dgl(nx_t, node_attrs, edge_attrs=None):
    g = dgl.DGLGraph()
    g.from_networkx(nx_t, node_attrs=node_attrs, edge_attrs=edge_attrs)

    return g