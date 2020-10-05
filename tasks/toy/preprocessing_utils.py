import os
import networkx as nx
from tqdm import tqdm
from preprocessing.base import Preprocessor
from preprocessing.utils import ConstValues
from utils.misc import eprint
from utils.serialization import to_pkl_file, from_pkl_file
from preprocessing.tree_conversions import string_to_nltk_tree, nltk_tree_to_nx


class ToyTreesPreprocessor(Preprocessor):

    def __init__(self, config):
        super(ToyTreesPreprocessor, self).__init__(config, typed=True)

    def preprocess(self):
        config = self.config
        input_dir = config.input_dir
        output_dir = config.output_dir

        # set file names
        file_names = {'train': 'train.txt',
                      'validation': 'dev.txt',
                      'test': 'test.txt'}

        with open(os.path.join(input_dir, 'vocabs.txt'), 'r') as f:
            for i, l in enumerate(f.readlines()):
                if i == 2:
                    self.types_vocab = {k: v for v,k in enumerate(l.strip().split('\t'))}

        # preprocessing trees
        for tag_name, fname in file_names.items():

            self.__init_stats__(tag_name)
            data_list = []

            with open(os.path.join(input_dir, fname), 'r') as f:
                for l in tqdm(f.readlines(), desc='Preprocessing {}'.format(tag_name)):
                    #ris, t = l.strip().split('\t')
                    t = l.strip()
                    nltk_t = string_to_nltk_tree(t)
                    nx_t = nltk_tree_to_nx(nltk_t,
                                           get_internal_node_dict=lambda w: {'x': ConstValues.NO_ELEMENT,
                                                                             'y': int(w.strip().split('_')[0]),
                                                                             't': self.types_vocab[w.strip().split('_')[1]]},
                                           get_leaf_node_dict=lambda w: {'x': int(w), 'y': ConstValues.NO_ELEMENT, 't': ConstValues.NO_ELEMENT},
                                           collapsePOS=False)

                    self.__update_stats__(tag_name, nx_t)
                    data_list.append((self.__nx_to_dgl__(nx_t)))

            self.__print_stats__(tag_name)
            to_pkl_file(data_list, os.path.join(output_dir, '{}.pkl'.format(tag_name)))

        # save all stats
        self.__save_stats__()