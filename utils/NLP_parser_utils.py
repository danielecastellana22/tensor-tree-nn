from nltk.parse.corenlp import CoreNLPDependencyParser, CoreNLPParser
from nltk import Tree
import networkx as nx

import re

from notebooks.notebook_utils import plot_netwrokx_tree
import matplotlib.pyplot as plt


def string_to_nltk_tree(s):
    NODE_PATTERN = '[^%s\t\n\r\f\v%s%s]+' % (re.escape(' '), re.escape('('), re.escape(')'))
    return Tree.fromstring(s, brackets='()', node_pattern=NODE_PATTERN, leaf_pattern=NODE_PATTERN)


class myParser:

    def get_tree(self, sentence):
        raise NotImplementedError


class myCoreNLPDepParser(CoreNLPDependencyParser, myParser):

    def make_tree(self, result):
        return myCoreNLPDepParser.transform_to_nx(result)

    @staticmethod
    def transform_to_nx(result):
        node_list = []
        edge_list = []
        for dependency in result['enhancedPlusPlusDependencies']:
            dependent_index = dependency['dependent']
            token = result['tokens'][dependent_index - 1]

            edge_list.append((dependent_index, dependency['governor'], {'type': dependency['dep']}))
            if dependent_index not in node_list:
                node_list.append((dependent_index, {
                    'word': token['word'],
                    'lemma': token['lemma'],
                    'tag': token['pos'],
                    'token_id': dependent_index})
                )

        g = nx.DiGraph()
        g.add_edges_from(edge_list)
        g.add_nodes_from(node_list)

        return g

    def get_tree(self, sentence):
        return self.raw_parse(sentence, properties={'tokenize.options': 'splitAssimilations=True'})


class myCoreNLPConstParser(CoreNLPParser, myParser):

    # we remove \s beacause \xa01 is used as non-separating white space
    # NODE_PATTERN = '[^%s%s%s]+' % (re.escape(' '), re.escape('('), re.escape(')'))
    # LEAF_PATTERN = '[^\n%s%s%s]' % (re.escape(' '), re.escape('('), re.escape(')'))

    @staticmethod
    def binarize(const_t_nltk:Tree):
        #collapse
        #Tree.collapse_unary(const_t_nltk, collapsePOS=True, collapseRoot=True)
        # chomsky normal form transformation
        Tree.chomsky_normal_form(const_t_nltk)

    @staticmethod
    def check_same_tokens(const_t: nx.DiGraph, dep_t: nx.DiGraph, bin_t: nx.DiGraph):
        sort_fun = lambda x: x[1]['token_id']
        const_leaves = sorted([(i, d) for i, d in const_t.nodes(data=True) if const_t.in_degree(i) == 0], key=sort_fun)
        dep_nodes = sorted([(i, d) for i, d in dep_t.nodes(data=True) if 'token_id' in d], key=sort_fun)
        bin_leaves = sorted([(i, d) for i, d in bin_t.nodes(data=True) if bin_t.in_degree(i) == 0], key=sort_fun)
        assert len(bin_leaves) == len(const_leaves)
        assert len(const_leaves) == len(dep_nodes)  # +1 to remove root node
        for i in range(len(const_leaves)):
            assert const_leaves[i][1]['token_id'] == dep_nodes[i][1]['token_id']
            assert const_leaves[i][1]['word'] == dep_nodes[i][1]['word']

            assert bin_leaves[i][1]['token_id'] == dep_nodes[i][1]['token_id']
            assert bin_leaves[i][1]['word'] == dep_nodes[i][1]['word']

    def make_tree(self, result):
        nltk_t = string_to_nltk_tree(result['parse'])
        const_t = self.tree_to_nxgraph(nltk_t)
        dep_t = myCoreNLPDepParser.transform_to_nx(result)
        nltk_t.chomsky_normal_form()
        bin_t = self.tree_to_nxgraph(nltk_t)
        myCoreNLPConstParser.check_same_tokens(const_t, dep_t, bin_t)
        return const_t, dep_t, bin_t

    def get_tree(self, sentence):
        return self.raw_parse(sentence, properties={})  # 'tokenize.options': 'strictTreebank3=true', 'parse.saveBinarized': 'true', splitAssimilations=true, splitHyphenated=true'})

    @staticmethod
    def tree_to_nxgraph(t:Tree):
        g = nx.DiGraph()
        token_id = 1

        def rec_parsing(node, pa_id):
            nonlocal g
            nonlocal token_id

            my_id = g.number_of_nodes()
            if isinstance(node, str):
                g.add_node(my_id, word=node, token_id=token_id)
                token_id += 1
            else:
                # internal node
                g.add_node(my_id, tag=node.label())

            if pa_id != -1:
                g.add_edge(my_id, pa_id)

            if not isinstance(node, str):
                for ch in node:
                    rec_parsing(ch, my_id)

        rec_parsing(t, -1)
        return g