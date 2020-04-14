from abc import abstractmethod
import networkx as nx
from utils.utils import eprint
from preprocessing.utils import ConstValues
from utils.serialization import to_json_file
import os


class Preprocessor:

    def __init__(self, config, typed):
        self.config = config
        self.stats = {}
        self.typed = typed
        if typed:
            self.types_stats = {}
            self.types_vocab = {}

    @abstractmethod
    def preprocess(self):
        raise NotImplementedError('This method must be implmented in a subclass!')

    def __init_stats__(self, tag_name):
        self.stats[tag_name] = {'tot_nodes': 0,
                                'tot_leaves': 0,
                                'no_labels': 0,
                                'max_out_degree': 0}
        if self.typed:
            self.stats[tag_name]['no_types'] = 0
            self.stats[tag_name]['num_types'] = 0
            self.types_stats[tag_name] = {}

    def __update_stats__(self, tag_name, t: nx.DiGraph):
        in_degree_list = [d for u, d in t.in_degree]

        self.stats[tag_name]['tot_nodes'] += t.number_of_nodes()
        self.stats[tag_name]['tot_leaves'] += len([x for x in in_degree_list if x == 0])
        self.stats[tag_name]['no_labels'] += len([i for i, d in t.nodes(data=True) if d['y'] == ConstValues.NO_ELEMENT])
        self.stats[tag_name]['max_out_degree'] = max(self.stats[tag_name]['max_out_degree'], max(in_degree_list))

        if self.typed:
            for i, d in t.nodes(data=True):
                t_id = d['type_id']
                if t_id != ConstValues.NO_ELEMENT:
                    if t_id not in self.types_stats[tag_name]:
                        self.types_stats[tag_name][t_id] = 0
                    self.types_stats[tag_name][t_id] += 1
                else:
                    self.stats[tag_name]['no_types'] += 1
            self.stats[tag_name]['num_types'] = len(self.types_stats[tag_name])

    def __print_stats__(self, tag_name):
        eprint('{} stats.'.format(tag_name))
        for k, v in self.stats[tag_name].items():
            eprint('{}:  {}.'.format(k, v))

    def __save_stats__(self):
        output_dir = self.config.output_dir
        if self.typed:
            eprint('Saving types stats.')
            rev_types_vocab = {v: k for k, v in self.types_vocab.items()}
            self.types_stats = {k: {rev_types_vocab[kk]: vv for kk, vv in v.items()} for k, v in self.types_stats.items()}
            to_json_file(self.types_stats, os.path.join(output_dir, 'types_stats.json'))
            eprint('Saving types vocabulary.')
            to_json_file(self.types_vocab, os.path.join(output_dir, 'types_vocab.json'))
        # save all stats
        eprint('Saving stats.')
        to_json_file(self.stats, os.path.join(output_dir, 'stats.json'))


