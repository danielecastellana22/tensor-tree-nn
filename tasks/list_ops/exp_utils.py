import os
import dgl
from tqdm import tqdm
from preprocessing.base import Preprocessor
from preprocessing.utils import ConstValues
from utils.serialization import to_pkl_file
from preprocessing.tree_conversions import string_to_nltk_tree, nltk_tree_to_nx
import torch as th
from experiments.base import CollateFun


class ListOpsPreprocessor(Preprocessor):

    def __init__(self, config):
        super(ListOpsPreprocessor, self).__init__(config, typed=True)

    def preprocess(self):
        config = self.config
        input_dir = config.input_dir
        output_dir = config.output_dir

        # set file names
        file_names = {'train': 'train.txt',
                      'validation': 'dev.txt',
                      'test': 'test.txt'}

        # preprocessing trees
        for tag_name, fname in file_names.items():

            self.__init_stats__(tag_name)
            data_list = []

            with open(os.path.join(input_dir, fname), 'r') as f:
                for l in tqdm(f.readlines(), desc='Preprocessing {}'.format(fname)):
                    ris, a = l.strip().split('\t')
                    if a[0] != '(':
                        a = '(' + a + ')'
                        
                    nx_a = nltk_tree_to_nx(string_to_nltk_tree(a),
                                           get_internal_node_dict=lambda w: {'x': ConstValues.NO_ELEMENT,
                                                                             'y': ConstValues.NO_ELEMENT,
                                                                             't': self.__get_type_id__(w.strip())},
                                           get_leaf_node_dict=lambda w: {'x': self.__get_word_id__(w.strip()),
                                                                         'y': ConstValues.NO_ELEMENT,
                                                                         't': ConstValues.NO_ELEMENT})

                    self.__update_stats__(tag_name, nx_a)
                    data_list.append((self.__nx_to_dgl__(nx_a), int(ris)))

            self.__print_stats__(tag_name)
            to_pkl_file(data_list, os.path.join(output_dir, '{}.pkl'.format(tag_name)))

        # save all stats
        self.__save_stats__()


class ListOpsCollateFun(CollateFun):

    def __call__(self, tuple_data):
        tree_list, out = zip(*tuple_data)
        batched_trees = dgl.batch(tree_list)
        batched_trees.to(self.device)

        out_tens = th.tensor(out, dtype=th.long)
        out_tens.to(self.device)

        return [batched_trees], out_tens
