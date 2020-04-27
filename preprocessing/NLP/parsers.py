from nltk.parse.corenlp import CoreNLPParser
from nltk import Tree
import networkx as nx
from preprocessing.tree_conversions import string_to_nltk_tree, nltk_tree_to_nx


class NLPAllParser(CoreNLPParser):

    def __init__(self):
        super().__init__()
        self.words_vocab = {}

    @staticmethod
    def __binarize__(const_t_nltk:Tree):
        # chomsky normal form transformation
        Tree.chomsky_normal_form(const_t_nltk)

    @staticmethod
    def __check_same_tokens__(const_t: nx.DiGraph, dep_t: nx.DiGraph, bin_t: nx.DiGraph):
        sort_fun = lambda x: x[1]['token_id']
        const_leaves = sorted([(i, d) for i, d in const_t.nodes(data=True) if const_t.in_degree(i) == 0], key=sort_fun)
        dep_nodes = sorted([(i, d) for i, d in dep_t.nodes(data=True) if 'token_id' in d], key=sort_fun)
        bin_leaves = sorted([(i, d) for i, d in bin_t.nodes(data=True) if bin_t.in_degree(i) == 0], key=sort_fun)
        assert len(bin_leaves) == len(const_leaves)
        assert len(const_leaves) == len(dep_nodes)  # +1 to remove root node
        for i in range(len(const_leaves)):
            assert const_leaves[i][1]['token_id'] == dep_nodes[i][1]['token_id']
            assert const_leaves[i][1]['word'].lower() == dep_nodes[i][1]['word'].lower()

            assert bin_leaves[i][1]['token_id'] == dep_nodes[i][1]['token_id']
            assert bin_leaves[i][1]['word'].lower() == dep_nodes[i][1]['word'].lower()

    def __update_words_vocab__(self, token_list):
        for x in token_list:
            k = x['word'].lower()
            id = len(self.words_vocab)
            self.words_vocab.setdefault(k, id)

    def make_tree(self, result):
        self.__update_words_vocab__(result['tokens'])
        nltk_t = string_to_nltk_tree(result['parse'])
        const_t = self.__const_tree_to_nx__(nltk_t)
        dep_t = self.__dep_graph_to_nx__(result)
        nltk_t.chomsky_normal_form()
        bin_t = self.__const_tree_to_nx__(nltk_t)
        self.__check_same_tokens__(const_t, dep_t, bin_t)
        return {'const': const_t, 'dep': dep_t, 'bin_const': bin_t}

    @staticmethod
    def __dep_graph_to_nx__(result):
        node_list = []
        edge_list = []
        for dependency in result['enhancedPlusPlusDependencies']:
            dependent_index = dependency['dependent']
            token = result['tokens'][dependent_index - 1]

            edge_list.append((dependent_index, dependency['governor'], {'type': dependency['dep']}))
            if dependent_index not in node_list:
                node_list.append((dependent_index, {
                    'word': token['word'].lower(),
                    'lemma': token['lemma'],
                    'tag': token['pos'],
                    'token_id': dependent_index})
                                 )

        g = nx.DiGraph()
        g.add_edges_from(edge_list)
        g.add_nodes_from(node_list)

        return g

    @staticmethod
    def __const_tree_to_nx__(t:Tree):
        return nltk_tree_to_nx(t,
                               get_internal_node_dict=lambda w: {'tag': w},
                               get_leaf_node_dict=lambda w: {'word': w})
