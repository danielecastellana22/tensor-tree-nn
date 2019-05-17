import torch.nn as nn
import torch as th

from treeLSTM import TreeLSTM, TreeDataset

import networkx as nx
import dgl
from collections import namedtuple
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

class MyTree:

    def __init__(self):
        self.child = []
        self.w = None


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

        t.child.insert(0,tl)

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
        t = MyTree()
        t.w = w

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
                tr = MyTree()
                tr.w = wr

                t.child.append(tr)

            # match closing bracket
            while s[idx] != ')':
                idx = idx+1
            idx = idx+1

            return t, idx

        else:
            new_t, idx = parse_string_tree(s, idx)

            t.child.append(new_t)

            # match closing bracket
            while s[idx] != ')':
                idx = idx+1
            idx = idx+1

            return t, idx


class LRTDataset(TreeDataset):

    LRTBatch = namedtuple('XORBatch', ['graph', 'mask', 'x', 'label'])

    NUM_CLASSES = 6
    NUM_VOCABS = 9

    def __init__(self, path_dir, file_name):
        TreeDataset.__init__(self, path_dir, file_name)

        self.__create_input_vocabulary()
        self.__create_output_vocabulary()
        self.__load_trees__()

    def __create_input_vocabulary(self):
        self.input_vocabulary = {}
        self.rev_input_vocabulary = []
        # add letter from a to f
        for i in range(6):
            ch = chr(ord('a') + i)
            self.rev_input_vocabulary.append(ch)
            self.input_vocabulary[ch] = len(self.input_vocabulary)

        # add operator and, or, not
        self.rev_input_vocabulary.append('and')
        self.input_vocabulary['and'] = len(self.input_vocabulary)
        self.rev_input_vocabulary.append('or')
        self.input_vocabulary['or'] = len(self.input_vocabulary)
        self.rev_input_vocabulary.append('not')
        self.input_vocabulary['not'] = len(self.input_vocabulary)

    def __create_output_vocabulary(self):
        self.output_vocabulary = {}
        self.rev_output_vocabulary = ['=', '>', '<', '^', '|', 'v', '#']
        for s in self.rev_output_vocabulary:
            self.output_vocabulary[s] = len(self.output_vocabulary)

    def __load_trees__(self):
        self.logger.debug('Loading trees.')
        # build trees
        with open(os.path.join(self.path_dir, self.file_name),'r') as txtfile:
            for sent in tqdm(txtfile.readlines(), desc='Loading trees: '):
                l_sent = sent[:-1].split('\t')
                symbol = l_sent[0]
                if len(l_sent[1]) == 1:
                    l_sent[1] = '( ' + l_sent[1] + ' )'
                if len(l_sent[2]) == 1:
                    l_sent[2] = '( ' + l_sent[2] + ' )'

                a_t, _ = parse_string_tree(l_sent[1], 0)
                a_t = self.__build_dgl_tree__(a_t)
                b_t, _ = parse_string_tree(l_sent[2], 0)
                b_t = self.__build_dgl_tree__(b_t)

                self.data.append((symbol, a_t, b_t))

        self.logger.info('{} trees loaded.'.format(len(self.data)))

    def __build_dgl_tree__(self, root):
        g = nx.DiGraph()

        def _rec_build(nid, node):

            for ch in node.child:
                cid = g.number_of_nodes()
                g.add_node(cid, x=self.input_vocabulary[ch.w], mask=0)
                _rec_build(cid, ch)

        # add root
        g.add_node(0, x=self.input_vocabulary[root.w], mask=0)
        _rec_build(0, root)
        ret = dgl.DGLGraph()
        ret.from_networkx(g, node_attrs=['x', 'mask'])
        return ret

    def get_loader(self, batch_size, device, shuffle=False):
        def batcher_dev(tuple_data):
            #tuple data contains (symbol,ltree,rtree)
            #TODO: find a way to batch trees correctly
            batched_trees = dgl.batch(tuple_data)
            return LRTDataset.LRTBatch(graph=batched_trees,
                                       mask=batched_trees.ndata['mask'].to(device),
                                       x=batched_trees.ndata['x'].to(device),
                                       label=2)

        return DataLoader(dataset=self, batch_size=batch_size, collate_fn=batcher_dev, shuffle=shuffle,
                          num_workers=0)


class LRTComparisonModule(nn.Module):

    def __init__(self, in_size, out_size):
        super(LRTComparisonModule, self).__init__()
        self.A = nn.Parameter(th.rand(in_size, in_size, out_size))
        self.U1 = nn.Linear(in_size, out_size, bias=False)
        self.U2 = nn.Linear(in_size, out_size, bias=False)
        self.b = nn.Parameter(th.rand(out_size))

        # neighbour_states has shape batch_size x n_neighbours x insize
        def forward(self, h_lsent, h_rsent):
            h_comb = th.einsum('ijk,ni,nj->nk', self.A, h_lsent, h_rsent) + self.U1(h_lsent) + self.U2(h_rsent) + self.b
            return th.tanh(h_comb)


class LRTModel(nn.Module):

    def __init__(self, x_size, h_size, cell_type='nary', **cell_args):
        super(LRTModel, self).__init__()
        self.tree_model = create_lrt_model(x_size, h_size, cell_type, **cell_args)
        self.comb_module = LRTComparisonModule(h_size, LRTDataset.NUM_CLASSES)

    def forward(self, l_trees_batched, r_trees_batched):
        h_ltree = self.tree_model(l_trees_batched)
        h_rtree = self.tree_model(r_trees_batched)


def create_lrt_model(x_size,
                     h_size,
                     cell_type='nary', **cell_args):

    num_vocabs = LRTDataset.NUM_VOCABS
    input_module = nn.Embedding(num_vocabs, x_size)

    output_module = nn.Identity()

    m = TreeLSTM(x_size, h_size, 2, input_module, output_module, cell_type, **cell_args)

    return m


def load_lrt_dataset(max_len_tree):
    #TODO: choose the number of operator
    #TODO: do split
    train_ds = LRTDataset('data/lrt', 'train1')
    dev_ds = LRTDataset('data/lrt', 'train2')
    test_ds = LRTDataset('data/lrt', 'test1')

    return train_ds, dev_ds, test_ds

def lrt_loss_function(output_model, true_label):
    aa = 4