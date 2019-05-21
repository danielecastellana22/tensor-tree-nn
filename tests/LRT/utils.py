import torch.nn as nn
import torch as th
import torch.nn.functional as F
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

    LRTBatch = namedtuple('LRTBatch', ['batch_a', 'batch_b', 'symbol'])

    NUM_CLASSES = 7
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
                symbol = self.output_vocabulary[l_sent[0]]
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
                g.add_node(cid, x=self.input_vocabulary[ch.w], y=-1, mask=1)
                _rec_build(cid, ch)
                g.add_edge(cid, nid)

            #assert (g.in_degree(nid) == 2 or g.in_degree(nid) == 0)


        # add root
        g.add_node(0, x=self.input_vocabulary[root.w], y=-1, mask=1)
        _rec_build(0, root)
        ret = dgl.DGLGraph()
        ret.from_networkx(g, node_attrs=['x', 'y', 'mask'])
        return ret

    def get_loader(self, batch_size, device, shuffle=False):
        def batcher_dev(tuple_data):
            #tuple data contains (symbol,ltree,rtree)
            symbol_list, graph_a_list, graph_b_list = zip(*tuple_data)
            batched_a_trees = dgl.batch(graph_a_list)
            batched_b_trees = dgl.batch(graph_b_list)
            tupled_batch_a = LRTDataset.TreeBatch(graph=batched_a_trees,
                                                  mask=batched_a_trees.ndata['mask'].to(device),
                                                  x=batched_a_trees.ndata['x'].to(device),
                                                  y=batched_a_trees.ndata['y'].to(device))

            tupled_batch_b = LRTDataset.TreeBatch(graph=batched_b_trees,
                                                  mask=batched_b_trees.ndata['mask'].to(device),
                                                  x=batched_b_trees.ndata['x'].to(device),
                                                  y=batched_a_trees.ndata['y'].to(device))

            return LRTDataset.LRTBatch(batch_a=tupled_batch_a,
                                       batch_b=tupled_batch_b,
                                       symbol=th.LongTensor(symbol_list).to(device))

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
        num_vocabs = LRTDataset.NUM_VOCABS
        input_module = nn.Embedding(num_vocabs, x_size)
        output_module = nn.Identity()
        self.tree_model = TreeLSTM(x_size, h_size, 2, input_module, output_module, cell_type, **cell_args)
        self.comb_module = LRTComparisonModule(h_size, LRTDataset.NUM_CLASSES)

    def forward(self, data_a, data_b):
        h_a_tree = self.tree_model(*data_a)
        h_b_tree = self.tree_model(*data_b)

        g_a = data_a[0]
        g_b = data_b[0]

        root_id_a = [i for i in range(g_a.number_of_nodes()) if g_a.out_degree(i) == 0]
        root_id_b = [i for i in range(g_b.number_of_nodes()) if g_b.out_degree(i) == 0]

        h_root_a = h_a_tree[root_id_a]
        h_root_b = h_b_tree[root_id_b]

        return self.comb_module(h_root_a, h_root_b)


def create_lrt_model(x_size, h_size, cell_type='nary', **cell_args):
    return LRTModel(x_size, h_size, cell_type, **cell_args)


def load_lrt_dataset(max_len_tree):
    #TODO: choose the number of operator

    train_ds = LRTDataset('data/lrt', 'train1')
    dev_ds = LRTDataset('data/lrt', 'train1')
    test_ds = LRTDataset('data/lrt', 'test1')

    return train_ds, dev_ds, test_ds


def lrt_loss_function(output_model, true_label):
    logp = F.log_softmax(output_model, 1)
    return F.nll_loss(logp, true_label, reduction='sum')


def lrt_extract_batch_data(batch):
    a_batched = batch.batch_a
    b_batched = batch.batch_a
    sym = batch.symbol

    g_a = a_batched.graph
    x_a = a_batched.x
    mask_a = a_batched.mask

    g_b = b_batched.graph
    x_b = b_batched.x
    mask_b = b_batched.mask

    return [[g_a, x_a, mask_a], [g_b, x_b, mask_b]], sym, None
