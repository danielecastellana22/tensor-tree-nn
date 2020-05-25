from abc import abstractmethod
import networkx as nx
from utils.utils import eprint, string2class
from preprocessing.utils import ConstValues
from preprocessing.tree_conversions import nx_to_dgl
from utils.serialization import to_json_file, from_pkl_file
import os


class Preprocessor:

    def __init__(self, config, typed):
        self.config = config
        self.tree_stats = {}
        self.out_degree_counting = {}
        self.leaves_counting = {}
        self.typed = typed
        self.words_vocab = {}
        if typed:
            self.types_counting = {}
            self.types_vocab = {}
            self.max_out_degree_for_type = {}

    @abstractmethod
    def preprocess(self):
        raise NotImplementedError('This method must be implmented in a subclass!')

    def __init_stats__(self, tag_name):
        self.tree_stats[tag_name] = {'tot_trees': 0,
                                     'tot_nodes': 0,
                                     'tot_leaves': 0,
                                     'no_labels': 0,
                                     'max_out_degree': 0}
        self.out_degree_counting[tag_name] = {}
        self.leaves_counting[tag_name] = {}

        if self.typed:
            self.tree_stats[tag_name]['no_types'] = 0
            self.tree_stats[tag_name]['num_types'] = 0
            self.types_counting[tag_name] = {}
            self.max_out_degree_for_type[tag_name] = {}

    def __update_stats__(self, tag_name, t: nx.DiGraph):
        in_degree_list = [d for u, d in t.in_degree]
        n_leaves = len([x for x in in_degree_list if x == 0])
        self.tree_stats[tag_name]['tot_trees'] += 1
        self.tree_stats[tag_name]['tot_nodes'] += t.number_of_nodes()
        self.tree_stats[tag_name]['tot_leaves'] += n_leaves
        self.tree_stats[tag_name]['no_labels'] += len([i for i, d in t.nodes(data=True) if d['y'] == ConstValues.NO_ELEMENT])
        self.tree_stats[tag_name]['max_out_degree'] = max(self.tree_stats[tag_name]['max_out_degree'], max(in_degree_list))
        # update out degree counting
        for x in in_degree_list:
            if x not in self.out_degree_counting[tag_name]:
                self.out_degree_counting[tag_name][x] = 0
            self.out_degree_counting[tag_name][x] += 1
        # update leaves counting
        if n_leaves not in self.leaves_counting[tag_name]:
            self.leaves_counting[tag_name][n_leaves] = 0
        self.leaves_counting[tag_name][n_leaves] += 1

        if self.typed:
            for i, d in t.nodes(data=True):
                t_id = d['t']
                if t_id != ConstValues.NO_ELEMENT:
                    if t_id not in self.types_counting[tag_name]:
                        self.types_counting[tag_name][t_id] = 0
                    self.types_counting[tag_name][t_id] += 1
                    if t_id not in self.max_out_degree_for_type[tag_name]:
                        self.max_out_degree_for_type[tag_name][t_id] = 0
                    self.max_out_degree_for_type[tag_name][t_id] = max(t.in_degree(i), self.max_out_degree_for_type[tag_name][t_id])
                else:
                    self.tree_stats[tag_name]['no_types'] += 1
            self.tree_stats[tag_name]['num_types'] = len(self.types_counting[tag_name])

    def __print_stats__(self, tag_name):
        eprint('{} stats:'.format(tag_name))
        for k, v in self.tree_stats[tag_name].items():
            eprint('{}:  {}.'.format(k, v))

    def __save_stats__(self):
        output_dir = self.config.output_dir
        if self.typed:
            eprint('Saving types stats.')
            rev_types_vocab = {v: k for k, v in self.types_vocab.items()}
            self.types_counting = {k: {rev_types_vocab[kk]: vv for kk, vv in v.items()} for k, v in self.types_counting.items()}
            self.max_out_degree_for_type = {k: {rev_types_vocab[kk]: vv for kk, vv in v.items()} for k, v in
                                   self.max_out_degree_for_type.items()}
            to_json_file(self.types_counting, os.path.join(output_dir, 'types_counting.json'))
            to_json_file(self.max_out_degree_for_type, os.path.join(output_dir, 'max_out_degree_for_type.json'))
            eprint('Saving types vocabulary.')
            to_json_file(self.types_vocab, os.path.join(output_dir, 'types_vocab.json'))
        # save all stats
        eprint('Saving stats.')
        to_json_file(self.words_vocab,os.path.join(output_dir, 'words_vocab.json'))
        to_json_file(self.tree_stats, os.path.join(output_dir, 'tree_stats.json'))
        to_json_file(self.out_degree_counting, os.path.join(output_dir, 'out_degree_counting.json'))
        to_json_file(self.leaves_counting, os.path.join(output_dir, 'leaves_counting.json'))

    def __get_type_id__(self, t):
        return self.types_vocab.setdefault(t, len(self.types_vocab))

    def __get_word_id__(self, w):
        return self.words_vocab.setdefault(w, len(self.words_vocab))

    def __nx_to_dgl__(self, t, other_attrs=None):

        if other_attrs is None:
            other_attrs = []

        all_attrs = ['x', 'y', 'x_mask'] + other_attrs

        if self.typed:
            all_attrs += ['t', 't_mask']

        return nx_to_dgl(t, node_attrs=all_attrs)


class NlpParsedTreesPreprocessor(Preprocessor):

    def __init__(self, config):
        tree_transformer_class = string2class(config.preprocessor_config.tree_transformer_class)
        super(NlpParsedTreesPreprocessor, self).__init__(config, tree_transformer_class.CREATE_TYPES)

        # create tree transformer
        if 'tree_transformer_params' in config.preprocessor_config:
            self.tree_transformer = tree_transformer_class(**config.preprocessor_config.tree_transformer_params)
        else:
            self.tree_transformer = tree_transformer_class()

        tree_type = config.preprocessor_config.tree_type
        if not isinstance(tree_type, list):
            config.preprocessor_config.tree_type = [tree_type]
        else:
            config.preprocessor_config.tree_type = list(sorted(tree_type))

        # load vocabulary
        eprint('Loading word vocabulary.')
        words_vocab_file = 'words_vocab.pkl'
        self.words_vocab = from_pkl_file(os.path.join(config.input_dir, words_vocab_file))

    def __get_word_id__(self, w):
        return self.words_vocab[w]

    @abstractmethod
    def preprocess(self):
        raise NotImplementedError('This method must be implmented in a subclass!')