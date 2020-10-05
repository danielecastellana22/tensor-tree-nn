from abc import abstractmethod
import networkx as nx
from utils.misc import eprint, string2class
from preprocessing.utils import ConstValues, load_embeddings
from preprocessing.tree_conversions import nx_to_dgl
from utils.serialization import to_json_file, from_pkl_file, to_pkl_file
import os


class Preprocessor:

    def __init__(self, config, typed):
        self.config = config
        self.dataset_stats = {}
        self.tree_stats = {}
        self.node_stats = {}
        self.typed = typed
        self.words_vocab = {}
        if typed:
            self.type_stats = {}
            self.types_vocab = {}

    @abstractmethod
    def preprocess(self):
        raise NotImplementedError('This method must be implmented in a subclass!')

    def __init_stats__(self, tag_name):
        self.dataset_stats[tag_name] = {'tot_trees': 0,
                                        'tot_nodes': 0,
                                        'tot_leaves': 0,
                                        'no_labels': 0,
                                        'max_out_degree': 0}
        self.tree_stats[tag_name] = {'num_leaves': [],
                                     'height': [],
                                     'max_out_degree': []}

        self.node_stats[tag_name] = {'out_degree': {},
                                     'depth': {}}

        if self.typed:
            self.dataset_stats[tag_name]['no_types'] = 0
            self.dataset_stats[tag_name]['num_types'] = 0
            self.type_stats[tag_name] = {'type_freq': {},
                                         'type_max_out_degree': {}}

    def __update_stats__(self, tag_name, t: nx.DiGraph):
        in_degree_list = [d for u, d in t.in_degree]
        n_leaves = len([x for x in in_degree_list if x == 0])
        rev_t = t.reverse()
        root = [u for u in t.nodes if t.out_degree(u) == 0][0]
        depth_list = {k:len(v)-1 for k,v in nx.shortest_path(rev_t, root).items()}

        # update dataset stats
        self.dataset_stats[tag_name]['tot_trees'] += 1
        self.dataset_stats[tag_name]['tot_nodes'] += t.number_of_nodes()
        self.dataset_stats[tag_name]['tot_leaves'] += n_leaves
        self.dataset_stats[tag_name]['no_labels'] += len([i for i, d in t.nodes(data=True) if d['y'] == ConstValues.NO_ELEMENT])
        self.dataset_stats[tag_name]['max_out_degree'] = max(self.dataset_stats[tag_name]['max_out_degree'], max(in_degree_list))

        # update tree stats
        self.tree_stats[tag_name]['num_leaves'].append(n_leaves)
        self.tree_stats[tag_name]['height'].append(max(depth_list.values()))
        self.tree_stats[tag_name]['max_out_degree'].append(max(in_degree_list))

        # update node stats
        for x in in_degree_list:
            if x not in self.node_stats[tag_name]['out_degree']:
                self.node_stats[tag_name]['out_degree'][x] = 0
            self.node_stats[tag_name]['out_degree'][x] += 1
        for x in depth_list.values():
            if x not in self.node_stats[tag_name]['depth']:
                self.node_stats[tag_name]['depth'][x] = 0
            self.node_stats[tag_name]['depth'][x] += 1

        # update type stats
        if self.typed:
            for i, d in t.nodes(data=True):
                t_id = d['t']
                if t_id != ConstValues.NO_ELEMENT:
                    # update freqeuncy
                    if t_id not in self.type_stats[tag_name]['type_freq']:
                        self.type_stats[tag_name]['type_freq'][t_id] = 0
                    self.type_stats[tag_name]['type_freq'][t_id] += 1

                    # update out degree
                    if t_id not in self.type_stats[tag_name]['type_max_out_degree']:
                        self.type_stats[tag_name]['type_max_out_degree'][t_id] = 0
                    self.type_stats[tag_name]['type_max_out_degree'][t_id] = max(t.in_degree(i), self.type_stats[tag_name]['type_max_out_degree'][t_id])
                else:
                    self.dataset_stats[tag_name]['no_types'] += 1
            self.dataset_stats[tag_name]['num_types'] = len(self.type_stats[tag_name]['type_freq'])

    def __print_stats__(self, tag_name):
        eprint('{} stats:'.format(tag_name))
        for k, v in self.dataset_stats[tag_name].items():
            eprint('{}:  {}.'.format(k, v))

    def __save_stats__(self):
        output_dir = self.config.output_dir
        if self.typed:
            eprint('Saving type stats.')
            rev_types_vocab = {v: k for k, v in self.types_vocab.items()}
            for k in self.type_stats:
                self.type_stats[k]['type_freq'] = {rev_types_vocab[kk]:vv for kk,vv in self.type_stats[k]['type_freq'].items()}
                self.type_stats[k]['type_max_out_degree'] = {rev_types_vocab[kk]: vv for kk, vv in self.type_stats[k]['type_max_out_degree'].items()}
            to_json_file(self.type_stats, os.path.join(output_dir, 'type_stats.json'))
            eprint('Saving types vocabulary.')
            to_json_file(self.types_vocab, os.path.join(output_dir, 'types_vocab.json'))

        # save all stats
        eprint('Saving stats.')
        to_json_file(self.words_vocab,os.path.join(output_dir, 'words_vocab.json'))
        to_json_file(self.dataset_stats, os.path.join(output_dir, 'dataset_stats.json'))
        to_json_file(self.tree_stats, os.path.join(output_dir, 'tree_stats.json'))
        to_json_file(self.node_stats, os.path.join(output_dir, 'node_stats.json'))

    def __get_type_id__(self, t):
        return self.types_vocab.setdefault(t, len(self.types_vocab))

    def __get_word_id__(self, w):
        return self.words_vocab.setdefault(w, len(self.words_vocab))

    def __nx_to_dgl__(self, t, other_node_attrs=None, other_edge_attrs=None):

        if other_node_attrs is None:
            other_node_attrs = []

        if other_edge_attrs is None:
            other_edge_attrs = []

        all_edge_attrs = other_edge_attrs

        all_node_attrs = ['x', 'y', 'pos'] + other_node_attrs
        if self.typed:
            all_node_attrs += ['t']

        if len(all_node_attrs) == 0:
            all_node_attrs = None

        if len(all_edge_attrs) == 0:
            all_edge_attrs = None

        return nx_to_dgl(t, node_attrs=all_node_attrs, edge_attrs=all_edge_attrs)


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

    def __save_word_embeddings__(self):
        eprint('Loading word embeddings.')
        pretrained_embs_file = self.config.pretrained_embs_file
        embedding_dim = self.config.embedding_dim
        pretrained_embs = load_embeddings(pretrained_embs_file, self.words_vocab, embedding_dim=embedding_dim)
        to_pkl_file(pretrained_embs, os.path.join(self.config.output_dir, 'pretrained_embs.pkl'))

        if 'type_pretrained_embs_file' in self.config:
            eprint('Loading type embeddings.')
            type_pretrained_embs = load_embeddings(self.config.type_pretrained_embs_file,
                                                   self.types_vocab,
                                                   embedding_dim=self.config.type_embedding_dim)
            to_pkl_file(type_pretrained_embs, os.path.join(self.config.output_dir, 'type_pretrained_embs.pkl'))

    @abstractmethod
    def preprocess(self):
        raise NotImplementedError('This method must be implmented in a subclass!')