import torch.nn as nn
import torch as th
import torch.nn.functional as F
from treeLSTM import TreeLSTM, TreeRNN, TreeDataset
import networkx as nx
import dgl
from collections import namedtuple
from torch.utils.data import DataLoader
from tqdm import tqdm
import os


class LogicalTree:

    def __init__(self):
        self.child = []
        self.w = None

    def check_tree(self):
        if not self.child:
            #is a leaf
            # must be a variable
            assert len(self.w) == 1
        else:
            # it is internal node
            # MUST BE AN OPERATOR
            assert len(self.w) > 1

            if self.w == 'not':
                assert len(self.child) == 1
            else:
                assert len(self.child) == 2

            for c in self.child:
                c.check_tree()



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

        t.child.insert(0, tl)

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
        t = LogicalTree()
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
                tr = LogicalTree()
                tr.w = wr

                t.child.append(tr)

            # match closing bracket
            while s[idx] != ')':
                idx = idx+1
            idx = idx+1

            return t, idx

        else:
            new_t, idx = parse_string_tree(s, idx)

            if len(t.w) == 1:
                # is a variable
                new_t.child.insert(0, t)
                t = new_t
            else:
                # is an operator
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

    def __init__(self, path_dir, file_name_list, name):
        TreeDataset.__init__(self, path_dir, file_name_list, name)

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
        f_names = [os.path.join(self.path_dir,x) for x in self.file_name_list]

        for f in f_names:
            with open(f, 'r') as txtfile:
                for sent in tqdm(txtfile.readlines(), desc='Loading trees: '):
                    l_sent = sent[:-1].split('\t')
                    symbol = self.output_vocabulary[l_sent[0]]
                    if len(l_sent[1]) == 1:
                        l_sent[1] = '( ' + l_sent[1] + ' )'
                    if len(l_sent[2]) == 1:
                        l_sent[2] = '( ' + l_sent[2] + ' )'

                    a_t, _ = parse_string_tree(l_sent[1], 0)
                    a_t.check_tree()
                    a_t = self.__build_dgl_tree__(a_t)
                    b_t, _ = parse_string_tree(l_sent[2], 0)
                    b_t.check_tree()
                    b_t = self.__build_dgl_tree__(b_t)

                    self.data.append((symbol, a_t, b_t))

        self.logger.info('{} trees loaded.'.format(len(self.data)))

    def __build_dgl_tree__(self, root):
        g = nx.DiGraph()
        vars_used = {}

        def _rec_build(nid, node):

            g.add_node(nid, x=self.input_vocabulary[node.w], y=-1, mask=1)

            if not node.child:
                # is a leaf
                vars_used[node.w] = 1

            for ch in node.child:
                cid = g.number_of_nodes()
                _rec_build(cid, ch)
                g.add_edge(cid, nid)

        _rec_build(0, root)
        ret = dgl.DGLGraph()
        ret.from_networkx(g, node_attrs=['x', 'y', 'mask'])

        n_vars = len(vars_used)
        #assert  n_vars <= 4
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
        #tz = th.zeros_like(h_comb)
        #return th.max(h_comb, tz) + 0.01* th.min(h_comb, tz)


class LRTModel(nn.Module):

    def __init__(self, x_size, h_size, cell_type='nary', **cell_args):
        super(LRTModel, self).__init__()
        num_vocabs = LRTDataset.NUM_VOCABS
        input_module = nn.Embedding(num_vocabs, x_size)

        #emb_matrix = th.zeros(num_vocabs,x_size, requires_grad=False)
        #emb_matrix[:num_vocabs-3, :] = th.randn(num_vocabs-3, x_size, requires_grad=False)*0.01
        #emb_matrix[num_vocabs-3, 0] = 1
        #emb_matrix[num_vocabs-2, 1] = 1
        #emb_matrix[num_vocabs-1, 2] = 1
        #input_module = nn.Embedding.from_pretrained(emb_matrix, freeze=True)

        output_module = nn.Identity()
        self.tree_model = TreeLSTM(x_size, h_size, 2, input_module, output_module, cell_type, **cell_args)
        #self.tree_model = TreeRNN(x_size, h_size, 2, input_module, output_module, cell_type, **cell_args)
        self.comb_module = LRTComparisonModule(h_size, LRTDataset.NUM_CLASSES)

    def forward(self, data_a, data_b):
        h_a_tree = self.tree_model(*data_a)
        h_b_tree = self.tree_model(*data_b)

        g_a = data_a[0]
        g_b = data_b[0]

        root_id_a = [i for i in range(g_a.number_of_nodes()) if g_a.out_degree(i) == 0]
        root_id_b = [i for i in range(g_b.number_of_nodes()) if g_b.out_degree(i) == 0]

        #a_list = dgl.unbatch(g_a)
        #b_list = dgl.unbatch(g_b)

        h_root_a = h_a_tree[root_id_a]
        h_root_b = h_b_tree[root_id_b]

        return self.comb_module(h_root_a, h_root_b)


def create_lrt_model(x_size, h_size, cell_type='nary', **cell_args):
    return LRTModel(x_size, h_size, cell_type, **cell_args)


def load_lrt_dataset(max_n_operator):
    max_n_operator=1
    tr_files = ['train' + str(x) for x in range(max_n_operator+1)]
    dev_files = ['dev' + str(x) for x in range(max_n_operator+1)]
    train_ds = LRTDataset('data/lrt', tr_files, name='train_set')
    dev_ds = LRTDataset('data/lrt', dev_files, name='dev_set')

    test_ds_list = []
    for x in range(1):
        ds = LRTDataset('data/lrt', ['test'+str(x)], name='test_set'+str(x))
        test_ds_list.append(ds)

    return train_ds, dev_ds, test_ds_list


def lrt_loss_function(output_model, true_label):
    #logp = F.log_softmax(output_model, 1)
    #return F.nll_loss(logp, true_label, reduction='sum')
    return F.cross_entropy(output_model, true_label, reduction='sum')


def lrt_extract_batch_data(batch):
    a_batched = batch.batch_a
    b_batched = batch.batch_b
    sym = batch.symbol

    g_a = a_batched.graph
    x_a = a_batched.x
    mask_a = a_batched.mask

    g_b = b_batched.graph
    x_b = b_batched.x
    mask_b = b_batched.mask

    n_batch = sym.size(0)

    return [[g_a, x_a, mask_a], [g_b, x_b, mask_b]], sym, n_batch, None
