from abc import abstractmethod
import networkx as nx
from utils.utils import eprint
from preprocessing.utils import ConstValues

class Preprocessor:

    def __init__(self, config):
        self.config = config
        self.stats = {}

    @abstractmethod
    def preprocess(self):
        raise NotImplementedError('This method must be implmented in a subclass!')

    def __init_stats__(self, tag_name):
        self.stats[tag_name] = {'tot_nodes': 0,
                                'tot_leaves': 0,
                                'no_labels': 0,
                                'max_out_degree': 0}

    def __update_stats__(self, tag_name, t:nx.DiGraph):
        in_degree_list = [d for u, d in t.in_degree]

        self.stats[tag_name]['tot_nodes'] += t.number_of_nodes()
        self.stats[tag_name]['tot_leaves'] += len([x for x in in_degree_list if x == 0])
        self.stats[tag_name]['no_labels'] += len([i for i, d in t.nodes(data=True) if d['y'] == ConstValues.NO_ELEMENT])
        self.stats[tag_name]['max_out_degree'] = max(self.stats[tag_name]['max_out_degree'], max(in_degree_list))

    def __print_stats__(self, tag_name):
        eprint('{} stats.'.format(tag_name))
        for k, v in self.stats[tag_name].items():
            eprint('{}:  {}.'.format(k, v))