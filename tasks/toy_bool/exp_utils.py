import os
import dgl
from tqdm import tqdm
from preprocessing.base import Preprocessor
from preprocessing.utils import ConstValues
from utils.serialization import to_pkl_file
from preprocessing.tree_conversions import string_to_nltk_tree, nltk_tree_to_nx
from experiments.base import CollateFun


class ToyBoolPreprocessor(Preprocessor):

    def __init__(self, config):
        super(ToyBoolPreprocessor, self).__init__(config, typed=True)

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

        self.words_vocab = {str(x): x for x in range(2)}

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
                                                                             't': self.__get_type_id__(w.strip().split('_')[1])},
                                           get_leaf_node_dict=lambda w: {'x': self.__get_word_id__(w), 'y': ConstValues.NO_ELEMENT, 't': ConstValues.NO_ELEMENT})

                    self.__update_stats__(tag_name, nx_t)
                    data_list.append((self.__nx_to_dgl__(nx_t)))

            self.__print_stats__(tag_name)
            to_pkl_file(data_list, os.path.join(output_dir, '{}.pkl'.format(tag_name)))

        # save all stats
        self.__save_stats__()


class ToyBoolCollateFun(CollateFun):

    def __init__(self, device, only_root=False):
        super(ToyBoolCollateFun, self).__init__(device)
        self.only_root = only_root

    def __call__(self, tuple_data):
        tree_list = tuple_data
        batched_trees = dgl.batch(tree_list)
        if not self.only_root:
            out = batched_trees.ndata['y']
        else:
            root_ids = [i for i in range(batched_trees.number_of_nodes()) if batched_trees.out_degree(i) == 0]
            out = batched_trees.ndata['y'][root_ids]

        batched_trees.to(self.device)
        out.to(self.device)

        return [batched_trees], out
