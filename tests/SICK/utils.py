import networkx as nx
import dgl
import torch.nn.functional as F
from nltk.corpus.reader import Tree
from collections import namedtuple, OrderedDict
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

from treeLSTM.cells import *
from treeLSTM.models import TreeLSTM
from treeLSTM.dataset import TreeDataset
from treeLSTM.metrics import MSE, Pearson


class SICKDataset(TreeDataset):

    PAD_WORD = -1  # special pad word id
    UNK_WORD = -1  # out-of-vocabulary word id

    SICKBatch = namedtuple('SICKBatch', ['batch_a', 'batch_b', 'score', 'target_distr'])

    NUM_CLASSES = 5

    def __init__(self, path_dir, file_name_list, vocab, name):
        TreeDataset.__init__(self, path_dir, file_name_list, name)
        self.max_out_degree = 0
        self.vocab = vocab
        self.__load_trees__()

    def __load_trees__(self):
        A_file_list = list(map(lambda x: x.replace('.txt', '_A.txt'), self.file_name_list))
        B_file_list = list(map(lambda x: x.replace('.txt', '_B.txt'), self.file_name_list))
        SCORE_file_list = list(map(lambda x: x.replace('.txt', '_SCORE.txt'), self.file_name_list))
        self.logger.debug('Loading trees.')

        tree_a = []
        with open(os.path.join(self.path_dir, A_file_list[0]), 'r') as f_A:
            for l in f_A.readlines():
                tree_a.append(Tree.fromstring(l))

        tree_b = []
        with open(os.path.join(self.path_dir, B_file_list[0]), 'r') as f_B:
            for l in f_B.readlines():
                tree_b.append(Tree.fromstring(l))

        with open(os.path.join(self.path_dir, SCORE_file_list[0]), 'r') as f_SCORE:
            for i,l in tqdm(enumerate(f_SCORE.readlines()), total=len(tree_a), desc='Loading trees: '):
                s = float(l[:-1])
                a_t = self.__build_dgl_tree__(tree_a[i])
                b_t = self.__build_dgl_tree__(tree_b[i])
                self.data.append((s, a_t, b_t))

        self.logger.info('{} trees loaded.'.format(len(self.data)))

    def get_loader(self, batch_size, device, shuffle=False):
        def get_target_distribution(labels):
            labels = np.array(labels)
            n_el = len(labels)
            target = np.zeros((n_el, SICKDataset.NUM_CLASSES))
            ceil = np.ceil(labels).astype(int)
            floor = np.floor(labels).astype(int)
            idx = (ceil == floor)
            not_idx = np.logical_not(idx)
            target[idx, floor[idx] - 1] = 1
            target[not_idx, floor[not_idx] - 1] = ceil[not_idx] - labels[not_idx]
            target[not_idx, ceil[not_idx] - 1] = labels[not_idx] - floor[not_idx]

            return target

        def batcher_dev(tuple_data):
            #tuple data contains (symbol,ltree,rtree)
            symbol_list, graph_a_list, graph_b_list = zip(*tuple_data)
            batched_a_trees = dgl.batch(graph_a_list)
            batched_b_trees = dgl.batch(graph_b_list)
            tupled_batch_a = SICKDataset.TreeBatch(graph=batched_a_trees,
                                                  mask=batched_a_trees.ndata['mask'].to(device),
                                                  x=batched_a_trees.ndata['x'].to(device),
                                                  y=batched_a_trees.ndata['y'].to(device))

            tupled_batch_b = SICKDataset.TreeBatch(graph=batched_b_trees,
                                                  mask=batched_b_trees.ndata['mask'].to(device),
                                                  x=batched_b_trees.ndata['x'].to(device),
                                                  y=batched_a_trees.ndata['y'].to(device))

            return SICKDataset.SICKBatch(batch_a=tupled_batch_a,
                                         batch_b=tupled_batch_b,
                                         score=th.FloatTensor(symbol_list).to(device),
                                         target_distr=th.FloatTensor(get_target_distribution(symbol_list)).to(device))

        return DataLoader(dataset=self, batch_size=batch_size, collate_fn=batcher_dev, shuffle=shuffle,
                          num_workers=0)

    def __build_dgl_tree__(self, root):
        g = nx.DiGraph()

        def _rec_build(nid, node):
            n_ch = len(node)
            if n_ch > self.max_out_degree:
                self.max_out_degree = n_ch

            for child in node:
                cid = g.number_of_nodes()
                if isinstance(child,str):
                    # leaf
                    w_id = self.vocab.get(child.lower(), self.UNK_WORD)
                    assert w_id != -1
                    g.add_node(cid, x=w_id, y=-1, mask=1)
                else:
                    # internal node
                    w_id = self.vocab.get(child.label().lower(), self.UNK_WORD)
                    assert w_id != -1
                    g.add_node(cid, x=w_id, y=-1, mask=1)
                    _rec_build(cid, child)
                g.add_edge(cid, nid)

        # add root
        w_id = self.vocab.get(root.label().lower(), self.UNK_WORD)
        assert w_id != -1
        g.add_node(0, x=w_id, y=-1, mask=1)
        _rec_build(0, root)
        ret = dgl.DGLGraph()
        ret.from_networkx(g, node_attrs=['x', 'y', 'mask'])
        return ret

    @property
    def num_vocabs(self):
        return len(self.vocab)


class SICKComparisonModule(nn.Module):

    def __init__(self, input_dim, num_classes):
        super(SICKComparisonModule, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = 50
        self.num_classes = num_classes
        self.wh = nn.Linear(2 * self.input_dim, self.hidden_dim)
        self.wp = nn.Linear(self.hidden_dim, self.num_classes)
        self.r = nn.Parameter(th.arange(1, num_classes+1).float().t(), requires_grad=False)

    def forward(self, lvec, rvec):
        mult_dist = th.mul(lvec, rvec)
        abs_dist = th.abs(th.add(lvec, -rvec))
        vec_dist = th.cat((mult_dist, abs_dist), 1)

        out = th.sigmoid(self.wh(vec_dist))
        out = F.log_softmax(self.wp(out), dim=1)
        pred = th.matmul(th.exp(out), self.r)

        return out, pred


class SICKModel(nn.Module):

    def __init__(self, x_size, h_size, cell, pretrained_emb=None, num_vocabs=-1):
        super(SICKModel, self).__init__()

        if pretrained_emb is None:
            input_module = nn.Embedding(num_vocabs, x_size)
        else:
            input_module = nn.Embedding.from_pretrained(pretrained_emb, freeze=True)

        output_module = nn.Identity()

        self.tree_model = TreeLSTM(x_size, h_size, input_module, output_module, cell)
        self.comb_module = SICKComparisonModule(h_size, SICKDataset.NUM_CLASSES)

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


def create_sick_model(x_size, h_size, max_output_degree, pretrained_emb=None, num_vocabs=None, cell_type='nary', rank=None, pos_stationarity=False):
    if cell_type == 'nary':
        cell = NaryCell(h_size, max_output_degree, pos_stationarity=pos_stationarity)
    elif cell_type == 'hosvd':
        cell = HOSVDCell(h_size, max_output_degree, rank=rank, pos_stationarity=pos_stationarity)
    elif cell_type == 'tt':
        cell = TTCell(h_size, max_output_degree, rank=rank, pos_stationarity=pos_stationarity)
    elif cell_type == 'cancomp':
        cell = CANCOMPCell(h_size, max_output_degree, rank=rank, pos_stationarity=pos_stationarity)
    elif cell_type == 'full':
        raise ValueError('The Full Tensora agrregation cannot be used')
    else:
        raise ValueError('Cell type not known')
    return SICKModel(x_size, h_size, cell, pretrained_emb, num_vocabs)


def load_sick_dataset(vocab):
    trainset = SICKDataset('data/sick/', ['SICK_train.txt'], vocab, name='train')
    devset = SICKDataset('data/sick/', ['SICK_trial.txt'], vocab, name='dev')
    testset = SICKDataset('data/sick/', ['SICK_test.txt'], vocab, name='test')
    return trainset, devset, testset


def sick_loss_function(output_model, true_label):
    return F.kl_div(output_model[0], true_label[0], reduction='batchmean')


def sick_extract_batch_data(batch):
    a_batched = batch.batch_a
    b_batched = batch.batch_b
    sym = batch.score
    target_distr = batch.target_distr

    g_a = a_batched.graph
    x_a = a_batched.x
    mask_a = a_batched.mask

    g_b = b_batched.graph
    x_b = b_batched.x
    mask_b = b_batched.mask

    n_batch = sym.size(0)

    return [[g_a, x_a, mask_a], [g_b, x_b, mask_b]], [target_distr, sym], n_batch, None


class MSE_sick(MSE):

    def update_metric(self, out, gold_label):
        super(MSE_sick, self).update_metric(out[1], gold_label[1])


class Pearson_sick(Pearson):

    def update_metric(self, out, gold_label):
        super(Pearson_sick, self).update_metric(out[1], gold_label[1])
