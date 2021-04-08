import os
import dgl
from tqdm import tqdm
from preprocessing.preprocessors import TreePreprocessor
from exputils.datasets import ConstValues
from exputils.serialisation import to_pkl_file
from preprocessing.tree_conversions import nltk_tree_to_nx
from nltk.tree import Tree
from exputils.datasets import CollateFun
import torch as th
import torch.nn as nn
import torch.nn.functional as F


def parse_string_tree(s, start):
    idx = start

    assert s[idx] == '('
    idx = idx+1

    while s[idx] == ' ':
        idx = idx+1

    if s[idx] == '(':

        tl, idx = parse_string_tree(s, idx)
        while s[idx] == ' ':
            idx = idx+1

        if s[idx] == '(':
            t, idx = parse_string_tree(s, idx)

        # is a leaf

        t.insert(0, tl)

        # match closing bracket
        while s[idx] != ')':
            idx = idx+1
        idx = idx+1

        return t, idx
    else:
        # there is an input element
        aux = idx+1
        while s[aux] != ' ' and s[aux] != ')' and s[aux] != '(':
            aux = aux+1
        w = s[idx:aux]
        idx = aux
        t = Tree(w, [])

        while s[idx] == ' ':
            idx = idx+1

        if s[idx] != '(':
            if s[idx] != ')':
                # another word
                aux = idx+1
                while s[aux] != ' ' and s[aux] != ')' and s[aux] != '(':
                    aux = aux + 1
                wr = s[idx:aux]
                idx = aux
                tr = Tree(wr, [])

                t.append(tr)

            # match closing bracket
            while s[idx] != ')':
                idx = idx+1
            idx = idx+1

            return t, idx

        else:
            new_t, idx = parse_string_tree(s, idx)

            if len(t.label()) == 1:
                # is a variable
                new_t.insert(0, t)
                t = new_t
            else:
                # is an operator
                t.append(new_t)

            # match closing bracket
            while s[idx] != ')':
                idx = idx+1
            idx = idx+1

            return t, idx


class LrtPreprocessor(TreePreprocessor):

    def __init__(self, config):
        super(LrtPreprocessor, self).__init__(config, typed=True)

    def preprocess(self):
        config = self.config
        input_dir = config.input_dir
        output_dir = config.output_dir
        train_max_num_ops = config.max_num_ops_in_training

        # set file names
        file_names = {'train': ['train{}'.format(x) for x in range(train_max_num_ops+1)],
                      'validation': ['dev{}'.format(x) for x in range(13)],
                      'test': ['test{}'.format(x) for x in range(13)]}

        # preprocessing trees
        for tag_name, fname_list in file_names.items():

            self.__init_stats__(tag_name)
            data_list = []

            for fname in fname_list:
                with open(os.path.join(input_dir, fname), 'r') as f:
                    for l in tqdm(f.readlines(), desc='Preprocessing {}'.format(fname)):
                        entail, a, b = l.strip().split('\t')

                        if a[0] != '(':
                            a = '(' + a + ')'

                        if b[0] != '(':
                            b = '(' + b + ')'
                        ax, _ = parse_string_tree(a, 0)
                        bx, _ = parse_string_tree(b, 0)
                        nx_a = nltk_tree_to_nx(ax,
                                               get_internal_node_dict=lambda w: {'x': ConstValues.NO_ELEMENT,
                                                                                 'y': ConstValues.NO_ELEMENT,
                                                                                 't': self.__get_type_id__(w.strip())},
                                               get_leaf_node_dict=lambda w: {'x': self.__get_word_id__(w.strip()),
                                                                             'y': ConstValues.NO_ELEMENT,
                                                                             't': ConstValues.NO_ELEMENT})

                        nx_b = nltk_tree_to_nx(bx,
                                               get_internal_node_dict=lambda w: {'x': ConstValues.NO_ELEMENT,
                                                                                 'y': ConstValues.NO_ELEMENT,
                                                                                 't': self.__get_type_id__(w.strip())},
                                               get_leaf_node_dict=lambda w: {'x': self.__get_word_id__(w.strip()),
                                                                             'y': ConstValues.NO_ELEMENT,
                                                                             't': ConstValues.NO_ELEMENT})

                        self.__update_stats__(tag_name, nx_a)
                        self.__update_stats__(tag_name, nx_b)
                        data_list.append((self.__nx_to_dgl__(nx_a), self.__nx_to_dgl__(nx_b), self.__get_output_id__(entail)))

            self.__print_stats__(tag_name)
            to_pkl_file(data_list, os.path.join(output_dir, '{}.pkl'.format(tag_name)))

        # save all stats
        self.__save_stats__()


class LrtCollateFun(CollateFun):

    def __init__(self, device, only_root=False):
        super(LrtCollateFun, self).__init__(device)
        self.only_root = only_root

    def __call__(self, tuple_data):
        a_tree_list, b_tree_list, out_list = zip(*tuple_data)

        batched_a_trees = dgl.batch(a_tree_list)
        batched_b_trees = dgl.batch(b_tree_list)

        batched_a_trees.to(self.device)
        batched_b_trees.to(self.device)

        out_tens = th.tensor(out_list, dtype=th.long)
        out_tens.to(self.device)

        return (batched_a_trees, batched_b_trees), out_tens


class LrtClassifier(nn.Module):

    def __init__(self, in_size, num_classes):
        super(LrtClassifier, self).__init__()
        self.A = nn.Parameter(th.rand(in_size, in_size, num_classes), requires_grad=True)
        self.U1 = nn.Linear(in_size, num_classes, bias=False)
        self.U2 = nn.Linear(in_size, num_classes, bias=False)
        self.b = nn.Parameter(th.rand(num_classes), requires_grad=True)

    # neighbour_states has shape batch_size x n_neighbours x insize
    def forward(self, h_lsent, h_rsent):
        h_comb = th.einsum('ijk,ni,nj->nk', self.A, h_lsent, h_rsent) + self.U1(h_lsent) + self.U2(h_rsent) + self.b
        return F.leaky_relu(h_comb, negative_slope=0.01)

