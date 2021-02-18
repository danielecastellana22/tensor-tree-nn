import os
import dgl
from tqdm import tqdm
from preprocessing.base import Preprocessor
from exputils.datasets import ConstValues
from exputils.serialisation import to_pkl_file
from preprocessing.tree_conversions import string_to_nltk_tree, nltk_tree_to_nx
from exputils.experiments import CollateFun
import networkx as nx


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

        self.words_vocab = {str(x): x for x in range(10)}

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
                    self.__add_intermediate_results__(nx_a)
                    assert not (nx_a.nodes[0]['y'] != int(ris) and nx_a.number_of_nodes()>1)
                    self.__update_stats__(tag_name, nx_a)
                    data_list.append((self.__nx_to_dgl__(nx_a)))

            self.__print_stats__(tag_name)
            to_pkl_file(data_list, os.path.join(output_dir, '{}.pkl'.format(tag_name)))

        # save all stats
        self.__save_stats__()

    def __get_op_fun__(self, op_id):
        import numpy as np
        rev_vocab = {v: k for k, v in self.types_vocab.items()}
        op_name = rev_vocab[op_id]
        op_fun = {
            'MAX': lambda x: int(np.max(x)),
            'MIN': lambda x: int(np.min(x)),
            'SM': lambda x: int(np.sum(x) % 10),
            'MED': lambda x: int(np.median(x))
        }

        return op_fun[op_name]

    def __add_intermediate_results__(self, t):

        for u in nx.topological_sort(t):
            if t.nodes[u]['t'] != ConstValues.NO_ELEMENT:
                op = t.nodes[u]['t']
                op_fun = self.__get_op_fun__(op)
                all_ch = list(t.predecessors(u))
                in_list = []
                for v in all_ch:
                    if t.nodes[v]['y'] != ConstValues.NO_ELEMENT:
                        in_list.append(t.nodes[v]['y'])
                    else:
                        in_list.append(t.nodes[v]['x'])
                t.nodes[u]['y'] = op_fun(in_list)


class ListOpsCollateFun(CollateFun):

    def __init__(self, device, only_root=False):
        super(ListOpsCollateFun, self).__init__(device)
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
