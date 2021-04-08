from nltk import Tree
import re
import dgl
import networkx as nx


def string_to_nltk_tree(s):
    node_pattern = '[^%s\t\n\r\f\v%s%s]+' % (re.escape(' '), re.escape('('), re.escape(')'))
    return Tree.fromstring(s, brackets='()', node_pattern=node_pattern, leaf_pattern=node_pattern)


def nx_to_dgl(nx_t, node_attrs, edge_attrs):
    g = dgl.from_networkx(nx_t, node_attrs=node_attrs, edge_attrs=edge_attrs)

    return g


def nltk_tree_to_nx(nltk_t, get_internal_node_dict, get_leaf_node_dict):
    g = nx.DiGraph()
    token_id = 1

    def _rec_parsing(node, pa_id):
        nonlocal g
        nonlocal token_id

        my_id = g.number_of_nodes()
        if isinstance(node, str):
            attr_dict = get_leaf_node_dict(node)
            attr_dict['token_id'] = token_id
            token_id += 1
        elif len(node) == 0:
            attr_dict = get_leaf_node_dict(node.label())
            attr_dict['token_id'] = token_id
            token_id += 1
        else:
            # internal node
            attr_dict = get_internal_node_dict(node.label())

        if pa_id != -1:
            attr_dict['pos'] = g.in_degrees(pa_id)
            g.add_node(my_id, **attr_dict)
            g.add_edge(my_id, pa_id)
        else:
            attr_dict['pos'] = -1
            g.add_node(my_id, **attr_dict)

        if not isinstance(node, str):
            for ch in node:
                _rec_parsing(ch, my_id)

    _rec_parsing(nltk_t, -1)
    return g