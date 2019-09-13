import torch.nn.functional as F
import pickle

from treeLSTM.models import TreeLSTM, TreeRNN
from treeLSTM.aggregators import SumChild, BinaryFullTensor, BaseAggregator
from treeLSTM.dataset import TreeDataset
from treeLSTM.metrics import Accuracy
from treeLSTM.trainer import *

from experiments.execution_utils import init_base_logger, get_base_logger
import networkx as nx
import dgl
from collections import namedtuple
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.init as INIT
import torch.optim as optim
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

    def get_variables(self):
        if len(self.child) == 0:
            return {self.w}
        else:
            ris = set([])
            for c in self.child:
                ris |= c.get_variables()
            return ris

    def count_operators(self):
        if len(self.child) == 0:
            return 0
        else:
            ris = 1
            for c in self.child:
                ris += c.count_operators()
            return ris


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


def parse_string_tree_infix(s, start):
    idx = start

    while s[idx] == ' ':
        idx = idx + 1

    if s[idx] == '(':
        t = LogicalTree()

        assert s[idx] == '('
        idx = idx + 1

        while s[idx] == ' ':
            idx = idx + 1



        tl, idx = parse_string_tree_infix(s, idx)
        t.child.append(tl)

        while s[idx] == ' ':
            idx = idx + 1

        tr, idx = parse_string_tree_infix(s, idx)
        t.child.append(tr)

        # match closing bracket
        while s[idx] != ')':
            idx = idx + 1
        idx = idx + 1

        return t, idx
    else:
        # is a leaf

        # there is an input element
        aux = idx
        while aux<len(s) and s[aux] != ' ' and s[aux] != ')' and s[aux] != '(':
            aux = aux + 1
        w = s[idx:aux]
        idx = aux

        t = LogicalTree()
        t.w = w

        while idx < len(s) and s[idx] == ' ':
            idx = idx + 1

        return t, idx


# TODO: think about a LRTDataset_base class
class LRTDataset(TreeDataset):

    LRTBatch = namedtuple('LRTBatch', ['batch_a', 'batch_b', 'symbol'])

    NUM_CLASSES = 7
    NUM_VOCABS = 9
    MAX_OUT_DEGREE = 2

    def __init__(self, path_dir, file_name_list, name):
        TreeDataset.__init__(self, path_dir, file_name_list, name)

        self.__create_input_vocabulary__()
        self.__create_output_vocabulary__()
        self.__load_trees__()

    def __create_input_vocabulary__(self):
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

    def __create_output_vocabulary__(self):
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
                    var_a = a_t.get_variables()

                    b_t, _ = parse_string_tree(l_sent[2], 0)
                    b_t.check_tree()
                    var_b = b_t.get_variables()

                    a_t = self.__build_dgl_tree__(a_t)
                    b_t = self.__build_dgl_tree__(b_t)
                    self.data.append((symbol, a_t, b_t))

        self.logger.info('{} trees loaded.'.format(len(self.data)))

    def __build_dgl_tree__(self, root):
        g = nx.DiGraph()

        def _rec_build(nid, node):

            g.add_node(nid, x=self.input_vocabulary[node.w], y=-1, mask=1)

            for ch in node.child:
                cid = g.number_of_nodes()
                _rec_build(cid, ch)
                g.add_edge(cid, nid)

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

    def merge_LRTdataset(self, other_ds):
        self.data = self.data + other_ds.data


class LRTDatasetTyped(LRTDataset):

    NUM_OPERATORS = 3

    def __init__(self, path_dir, file_name_list, name):
        LRTDataset.__init__(self, path_dir, file_name_list, name)
        self.__create_input_vocabulary__()

    def __create_input_vocabulary__(self):
        self.input_vocabulary = {}
        self.rev_input_vocabulary = []
        # add letter from a to f
        for i in range(6):
            ch = chr(ord('a') + i)
            self.rev_input_vocabulary.append(ch)
            self.input_vocabulary[ch] = len(self.input_vocabulary)

        self.op_vocabulary = {'and': 0, 'or': 1, 'not': 2}

    def __build_dgl_tree__(self, root):
        g = nx.DiGraph()

        def _rec_build(nid, node):

            if not node.child:
                # is a leaf
                g.add_node(nid, type=-1, x=self.input_vocabulary[node.w], y=-1, mask=1)
            else:
                # is internal
                g.add_node(nid, type=self.op_vocabulary[node.w], x=self.op_vocabulary[node.w], y=-1, mask=0)

            for ch in node.child:
                cid = g.number_of_nodes()
                _rec_build(cid, ch)
                g.add_edge(cid, nid)

        _rec_build(0, root)
        ret = dgl.DGLGraph()
        ret.from_networkx(g, node_attrs=['x', 'y', 'mask', 'type'])

        return ret


class LRTDatasetInfix(TreeDataset):

    LRTBatch = namedtuple('LRTBatch', ['batch_a', 'batch_b', 'symbol'])

    NUM_CLASSES = 7
    NUM_VOCABS = 9
    MAX_OUT_DEGREE = 2

    def __init__(self, path_dir, file_name_list, name):
        TreeDataset.__init__(self, path_dir, file_name_list, name)

        self.__create_input_vocabulary__()
        self.__create_output_vocabulary__()
        self.__load_trees__()

    def __create_input_vocabulary__(self):
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

    def __create_output_vocabulary__(self):
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

                    # if len(l_sent[1]) == 1:
                    #     l_sent[1] = '( ' + l_sent[1] + ' )'
                    # if len(l_sent[2]) == 1:
                    #     l_sent[2] = '( ' + l_sent[2] + ' )'

                    a_t, _ = parse_string_tree_infix(l_sent[1], 0)

                    b_t, _ = parse_string_tree_infix(l_sent[2], 0)

                    a_t = self.__build_dgl_tree__(a_t)
                    b_t = self.__build_dgl_tree__(b_t)
                    self.data.append((symbol, a_t, b_t))

        self.logger.info('{} trees loaded.'.format(len(self.data)))

    def __build_dgl_tree__(self, root):
        g = nx.DiGraph()

        def _rec_build(nid, node):

            if node.w is not None:
                g.add_node(nid, x=self.input_vocabulary[node.w], y=-1, mask=1)
            else:
                g.add_node(nid, x=-1, y=-1, mask=0)

            for ch in node.child:
                cid = g.number_of_nodes()
                _rec_build(cid, ch)
                g.add_edge(cid, nid)

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

    def merge_LRTdataset(self, other_ds):
        self.data = self.data + other_ds.data


class TypedAggregator(BaseAggregator):

    def __init__(self, h_size, max_output_degree, pos_stationarity, n_aggr, **kwargs):
        if max_output_degree > 2:
            raise ValueError('Full cel type can be use only with a maximum output degree of 2')

        super(TypedAggregator, self).__init__(h_size, max_output_degree, pos_stationarity, n_aggr)

        self.n_type = kwargs['n_type']
        self.cell_list = nn.ModuleList()
        for i in range(self.n_type):
            self.cell_list.append(kwargs['agg_class'](h_size, max_output_degree, pos_stationarity, n_aggr, **kwargs))

    def forward(self, neighbour_h, nodes):

        # get type
        ris = th.zeros((neighbour_h.size(0), self.n_aggr*neighbour_h.size(2)))
        for i in range(self.n_type):
            mask = nodes.data['type'] == i
            if th.sum(mask) > 0:
                ris[mask, :] = self.cell_list[i](neighbour_h[mask, :, :], nodes)

        return ris


# TODO: merge followin 3 functions adding a parameter
def load_LRT_dataset(max_n_operator, load_test=True):
    data_dir = "data/lrt"
    train_obj_file = os.path.join(data_dir, 'train_{}.pkl'.format(max_n_operator))
    dev_obj_file = os.path.join(data_dir, 'val.pkl')
    test_obj_file = os.path.join(data_dir, 'test.pkl')

    # training set
    if os.path.exists(train_obj_file):
        with open(train_obj_file, 'rb') as f:
            trainset = pickle.load(f)
        get_sub_logger('train').info('Training set has already parsed.')
    else:
        tr_files = ['train' + str(x) for x in range(max_n_operator + 1)]
        trainset = LRTDataset(data_dir, tr_files, name='train_set')
        with open(train_obj_file, 'wb') as f:
            pickle.dump(trainset, f)

    # validation set
    if os.path.exists(dev_obj_file):
        with open(dev_obj_file, 'rb') as f:
            devset_list = pickle.load(f)
        get_sub_logger('validation').info('Validation set has already parsed.')
    else:
        devset_list = []
        for x in range(12):
            dev_ds = LRTDataset(data_dir, ['dev' + str(x)], name='dev_set' + str(x))
            devset_list.append(dev_ds)

        with open(dev_obj_file, 'wb') as f:
            pickle.dump(devset_list, f)

    testset_list = []
    if load_test:
        # test set
        if os.path.exists(test_obj_file):
            with open(test_obj_file, 'rb') as f:
                testset_list = pickle.load(f)
            get_sub_logger('test').info('Test set has already parsed.')
        else:
            testset_list = []
            for x in range(12):
                test_ds = LRTDataset(data_dir, ['test'+str(x)], name='test_set'+str(x))
                testset_list.append(test_ds)

            with open(test_obj_file, 'wb') as f:
                pickle.dump(testset_list, f)

    return trainset, devset_list, testset_list


def load_LRT_dataset_infix(max_n_operator, load_test=True):
    data_dir = "data/lrt"
    train_obj_file = os.path.join(data_dir, 'train_{}_infix.pkl'.format(max_n_operator))
    dev_obj_file = os.path.join(data_dir, 'val_infix.pkl')
    test_obj_file = os.path.join(data_dir, 'test_infix.pkl')

    # training set
    if os.path.exists(train_obj_file):
        with open(train_obj_file, 'rb') as f:
            trainset = pickle.load(f)
        get_sub_logger('train').info('Training set has already parsed.')
    else:
        tr_files = ['train' + str(x) for x in range(max_n_operator + 1)]
        trainset = LRTDatasetInfix(data_dir, tr_files, name='train_set')
        with open(train_obj_file, 'wb') as f:
            pickle.dump(trainset, f)

    # validation set
    if os.path.exists(dev_obj_file):
        with open(dev_obj_file, 'rb') as f:
            devset_list = pickle.load(f)
        get_sub_logger('validation').info('Validation set has already parsed.')
    else:
        devset_list = []
        for x in range(12):
            dev_ds = LRTDatasetInfix(data_dir, ['dev' + str(x)], name='dev_set' + str(x))
            devset_list.append(dev_ds)

        with open(dev_obj_file, 'wb') as f:
            pickle.dump(devset_list, f)

    testset_list = []
    if load_test:
        # test set
        if os.path.exists(test_obj_file):
            with open(test_obj_file, 'rb') as f:
                testset_list = pickle.load(f)
            get_sub_logger('test').info('Test set has already parsed.')
        else:
            testset_list = []
            for x in range(12):
                test_ds = LRTDatasetInfix(data_dir, ['test'+str(x)], name='test_set'+str(x))
                testset_list.append(test_ds)

            with open(test_obj_file, 'wb') as f:
                pickle.dump(testset_list, f)

    return trainset, devset_list, testset_list


def load_LRT_dataset_typed(max_n_operator, load_test=True):
    data_dir = "data/lrt"
    train_obj_file = os.path.join(data_dir, 'train_{}_typed.pkl'.format(max_n_operator))
    dev_obj_file = os.path.join(data_dir, 'val_typed.pkl')
    test_obj_file = os.path.join(data_dir, 'test_typed.pkl')

    # training set
    if os.path.exists(train_obj_file):
        with open(train_obj_file, 'rb') as f:
            trainset = pickle.load(f)
        get_sub_logger('train').info('Training set has already parsed.')
    else:
        tr_files = ['train' + str(x) for x in range(max_n_operator + 1)]
        trainset = LRTDatasetTyped(data_dir, tr_files, name='train_set')
        with open(train_obj_file, 'wb') as f:
            pickle.dump(trainset, f)

    # validation set
    if os.path.exists(dev_obj_file):
        with open(dev_obj_file, 'rb') as f:
            devset_list = pickle.load(f)
        get_sub_logger('validation').info('Validation set has already parsed.')
    else:
        devset_list = []
        for x in range(12):
            dev_ds = LRTDatasetTyped(data_dir, ['dev' + str(x)], name='dev_set' + str(x))
            devset_list.append(dev_ds)

        with open(dev_obj_file, 'wb') as f:
            pickle.dump(devset_list, f)

    # test set
    testset_list = []
    if load_test:
        if os.path.exists(test_obj_file):
            with open(test_obj_file, 'rb') as f:
                testset_list = pickle.load(f)
            get_sub_logger('test').info('Test set has already parsed.')
        else:
            testset_list = []
            for x in range(12):
                test_ds = LRTDatasetTyped(data_dir, ['test'+str(x)], name='test_set'+str(x))
                testset_list.append(test_ds)

            with open(test_obj_file, 'wb') as f:
                pickle.dump(testset_list, f)

    return trainset, devset_list, testset_list


def LRT_loss_function(output_model, true_label):
    return F.cross_entropy(output_model, true_label, reduction='sum')


def extract_LRT_batch_data(batch):
    a_batched = batch.batch_a
    b_batched = batch.batch_b
    sym = batch.symbol

    g_a = a_batched.graph
    x_a = a_batched.x
    mask_a = a_batched.mask

    g_b = b_batched.graph
    x_b = b_batched.x
    mask_b = b_batched.mask

    return [[g_a, x_a, mask_a], [g_b, x_b, mask_b]], sym, None


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
        #return th.tanh(h_comb)
        tz = th.zeros_like(h_comb)
        return th.max(h_comb, tz) + 0.01* th.min(h_comb, tz)


class LRTModel(nn.Module):

    def __init__(self,use_one_hot_encoding,
                 tree_model_class, x_size, h_size, pos_stationarity,
                aggregator_class, **kwargs):
        super(LRTModel, self).__init__()
        num_vocabs = LRTDataset.NUM_VOCABS
        if not use_one_hot_encoding:
            input_module = nn.Embedding(num_vocabs, x_size)
        else:
            input_module = nn.Embedding.from_pretrained(th.eye(LRTDataset.NUM_VOCABS), freeze=True)

        output_module = nn.Identity()

        self.tree_model = tree_model_class(x_size, h_size, LRTDataset.MAX_OUT_DEGREE, pos_stationarity, input_module, output_module,
                                           aggregator_class, **kwargs)

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


def create_LRT_model(use_one_hot_encoding, typed,
                     tree_model_type, x_size, h_size, pos_stationarity,
                     cell_type, rank):

    if tree_model_type == 'treeLSTM':
        tree_model_class = TreeLSTM
    else:
        tree_model_class = TreeRNN

    if cell_type == 'sumchild':
            agg_class = SumChild
    elif cell_type == 'full':
            agg_class = BinaryFullTensor
    else:
        raise ValueError('Cell type not known')

    if typed:
        return LRTModel(use_one_hot_encoding,
                        tree_model_class, x_size, h_size, pos_stationarity,
                        TypedAggregator, n_type=LRTDatasetTyped.NUM_OPERATORS, agg_class=agg_class, rank=rank)
    else:
        return LRTModel(use_one_hot_encoding,
                        tree_model_class, x_size, h_size, pos_stationarity,
                        agg_class, rank=rank)


def LRT_single_run_fun(args, device, log_dir):

    if not hasattr(args,'use_one_hot'):
        args.use_one_hot = False

    if args.use_one_hot:
        args.x_size = LRTDataset.NUM_VOCABS

    if args.typed_data and args.infix_data:
        raise ValueError('You can specify either typed or infix data')

    if args.typed_data:
        # load the data
        trainset, devset_list, testset_list = load_LRT_dataset_typed(max_n_operator=4)
    elif args.infix_data:
        trainset, devset_list, testset_list = load_LRT_dataset_infix(max_n_operator=4)
    else:
        # load the data
        trainset, devset_list, testset_list = load_LRT_dataset(max_n_operator=4)

    conc_devset = devset_list[0]
    for i in range(1, len(devset_list)):
        conc_devset.merge_LRTdataset(devset_list[i])

    # create the model
    model = create_LRT_model(args.use_one_hot, args.typed_data,
                             args.tree_model, args.x_size, args.h_size, args.pos_stationarity,
                             args.cell_type, rank=args.rank).to(device)

    # create the optimizser
    params_no_cell = [x[1] for x in list(model.named_parameters()) if
                      x[1].requires_grad and 'cell' not in x[0]]
    params_cell = [x[1] for x in list(model.named_parameters()) if
                   x[1].requires_grad and 'cell' in x[0]]

    for p in params_cell:
        if p.dim() > 1:
            INIT.kaiming_normal_(p)

    for p in params_no_cell:
        if p.dim() > 1:
            INIT.kaiming_normal_(p)

    # create the optimizer
    #optimizer = optim.Adagrad([
    #    {'params': params_no_cell, 'lr': args.lr, 'weight_decay': args.weight_decay},
    #    {'params': params_cell, 'lr': args.lr, 'weight_decay': args.weight_decay}])

    optimizer = optim.Adadelta([
        {'params': params_no_cell, 'weight_decay': args.weight_decay},
        {'params': params_cell, 'weight_decay': args.weight_decay}])

    # train and validate
    best_model, info_training = train_and_validate(model, extract_LRT_batch_data, LRT_loss_function, optimizer, trainset,
                                                   conc_devset, device,
                                                   metrics_class=[Accuracy],
                                                   batch_size=args.batch_size,
                                                   n_epochs=args.epochs, early_stopping_patience=args.early_stopping,
                                                   evaluate_on_training_set=True)

    th.save(best_model, os.path.join(log_dir, 'best_model.pkl'))
    th.save(info_training, os.path.join(log_dir, 'info_training.pkl'))

    # test
    test_metrics = []
    test_predictions = []
    for i in range(len(testset_list)):
        tm, tp = test(best_model, extract_LRT_batch_data, testset_list[i], device,
                      metrics_class=[Accuracy],
                      batch_size=args.batch_size)
        test_metrics.append(tm)
        test_predictions.append(tp)

    th.save({'test_metrics': test_metrics, 'test_predictions': test_predictions},
            os.path.join(log_dir, 'test_results.pkl'))


def get_LRT_model_selection_fun(args, device):

    if not hasattr(args, 'use_one_hot'):
        args.use_one_hot = False

    if args.use_one_hot:
        args.x_size = LRTDataset.NUM_VOCABS

    if args.typed_data and args.infix_data:
        raise ValueError('You can specify either typed or infix data')

    def train_foo(id, log_dir, params):
        init_base_logger(log_dir, 'exp{}'.format(id))
        logger = get_base_logger()
        logger.info(str(params))

        if args.typed_data:
            # load the data
            trainset, devset_list, _ = load_LRT_dataset_typed(max_n_operator=4, load_test=False)
        elif args.infix_data:
            trainset, devset_list, _ = load_LRT_dataset_infix(max_n_operator=4, load_test=False)
        else:
            # load the data
            trainset, devset_list, _ = load_LRT_dataset(max_n_operator=4, load_test=False)

        conc_devset = devset_list[0]
        for i in range(1, len(devset_list)):
            conc_devset.merge_LRTdataset(devset_list[i])

        # create the model
        model = create_LRT_model(params['x_size'], params['h_size'], trainset.NUM_CLASSES,
                                 use_one_hot_encoding=args.use_one_hot,
                                 max_output_degree=trainset.MAX_OUT_DEGREE,
                                 cell_type=args.cell_type,
                                 rank=params['rank'],
                                 pos_stationarity=args.pos_stationarity,
                                 scale_factor=args.scale_factor,
                                 typed=args.typed_data).to(device)


        # create the optimizser
        params_no_cell = [x[1] for x in list(model.named_parameters()) if
                          x[1].requires_grad and 'cell' not in x[0]]
        params_cell = [x[1] for x in list(model.named_parameters()) if
                       x[1].requires_grad and 'cell' in x[0]]

        for p in params_cell:
            if p.dim() > 1:
                INIT.kaiming_normal_(p)

        for p in params_no_cell:
            if p.dim() > 1:
                INIT.kaiming_normal_(p)

        # create the optimizer
        optimizer = optim.Adagrad([
            {'params': params_no_cell, 'lr': params['lr'], 'weight_decay': params['weight_decay']},
            {'params': params_cell, 'lr': params['lr'], 'weight_decay': params['weight_decay']}])

        # train and validate
        best_model, info_training = train_and_validate(model, extract_LRT_batch_data, LRT_loss_function, optimizer,
                                                       trainset,
                                                       conc_devset, device,
                                                       metrics_class=[Accuracy],
                                                       batch_size=args.batch_size,
                                                       n_epochs=args.epochs,
                                                       early_stopping_patience=args.early_stopping,
                                                       evaluate_on_training_set=True)

        th.save(best_model, os.path.join(log_dir, 'best_model.pkl'))
        th.save(info_training, os.path.join(log_dir, 'info_training.pkl'))

        dev_metrics = []
        for i in range(len(devset_list)):
            tm, _ = test(best_model, extract_LRT_batch_data, devset_list[i], device,
                          metrics_class=[Accuracy],
                          batch_size=args.batch_size)
            dev_metrics.append(tm)

        th.save(dev_metrics, os.path.join(log_dir, 'dev_metrics.pkl'))

        ris = {}
        best_epoch = info_training['best_epoch']
        ris['tr_acc'] = info_training['tr_metrics'][Accuracy.__name__][best_epoch]
        ris['dev_acc'] = info_training['dev_metrics'][Accuracy.__name__][best_epoch]
        #ris['dev_acc_length'] = dev_metrics

        return ris

    return train_foo

