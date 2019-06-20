import torch.nn as nn
import torch.nn.functional as F
from treeLSTM.models import TreeLSTM
from treeLSTM.dataset import TreeDataset
from treeLSTM.cells import *
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
from nltk import Tree
import dgl
import networkx as nx


class HTENSDataset(TreeDataset):

    def __init__(self, path_dir, file_name_list, name):
        TreeDataset.__init__(self, path_dir, file_name_list, name)
        self.__load_trees__()

    def __load_trees__(self):
        self.logger.debug('Loading trees.')
        # build trees
        name_list = [os.path.join(self.path_dir, x) for x in self.file_name_list]
        for f_name in name_list:
            with open(f_name,'r') as txtfile:
                for sent in tqdm(txtfile.readlines(), desc='Loading trees: '):
                    t = Tree.fromstring(sent)
                    self.data.append(self.__build_dgl_tree__(t))

        self.logger.info('{} trees loaded.'.format(len(self.data)))

    def get_loader(self, batch_size, device, shuffle=False):
        def batcher_dev(batch):
            batched_trees = dgl.batch(batch)
            return HTENSDataset.TreeBatch(graph=batched_trees,
                                          mask=batched_trees.ndata['mask'].to(device),
                                          x=batched_trees.ndata['x'].to(device),
                                          y=batched_trees.ndata['y'].to(device))

        return DataLoader(dataset=self, batch_size=batch_size, collate_fn=batcher_dev, shuffle=shuffle,
                          num_workers=0)


    def __build_dgl_tree__(self, root):
        g = nx.DiGraph()

        def _rec_build(nid, node):
            for child in node:
                cid = g.number_of_nodes()
                if isinstance(child[0], str) or isinstance(child[0], bytes):
                    # leaf node
                    word = 0 if child[0].lower() == 'a' else 1
                    g.add_node(cid, x=word, y=int(child.label()), mask=1)
                else:
                    g.add_node(cid, x=-1, y=int(child.label()), mask=0)
                    _rec_build(cid, child)
                g.add_edge(cid, nid)

        # add root
        g.add_node(0, x=-1, y=int(root.label()), mask=0)
        _rec_build(0, root)
        ret = dgl.DGLGraph()
        ret.from_networkx(g, node_attrs=['x', 'y', 'mask'])
        return ret


class HTENSOutputModule(nn.Module):

    def __init__(self,h_size, num_classes, dropout):
        super(HTENSOutputModule, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(h_size, num_classes)

    def forward(self, h):
        return self.linear(self.dropout(h))


def create_htens_model(x_size, h_size, dropout, cell_type='nary', rank=None, pos_stationarity=False):
    max_output_degree = 2

    if cell_type == 'nary':
        cell = NaryCell(h_size, max_output_degree, pos_stationarity=pos_stationarity)
    elif cell_type == 'hosvd':
        cell = HOSVDCell(h_size, max_output_degree, rank=rank, pos_stationarity=pos_stationarity)
    elif cell_type == 'tt':
        cell = TTCell(h_size, max_output_degree, rank=rank)
    elif cell_type == 'cancomp':
        cell = CANCOMPCell(h_size, max_output_degree, rank=rank, pos_stationarity=pos_stationarity)
    elif cell_type == 'full':
        cell = BinaryFullTensorCell(h_size, max_output_degree, pos_stationarity)
    else:
        raise ValueError('Cell type not known')

    num_classes = 2
    num_vocabs = 2
    input_module = nn.Embedding(num_vocabs, x_size)

    output_module = HTENSOutputModule(h_size, num_classes, dropout)

    return TreeLSTM(x_size, h_size, input_module, output_module, cell)
    #m = TreeLSTM(x_size, h_size, 2, input_module, output_module, cell_type, **cell_args)


def load_htens_dataset():
    trainset = HTENSDataset('data/htens/', ['train.txt'], name='train')
    devset = HTENSDataset('data/htens/', ['dev.txt'], name='dev')
    testset = HTENSDataset('data/htens/', ['test.txt'], name='test')

    return trainset, devset, testset


def htens_loss_function(output_model, true_label):
    logp = F.log_softmax(output_model, 1)
    return F.nll_loss(logp, true_label, reduction='sum')


def htens_extract_batch_data(batch):
    g = batch.graph
    x = batch.x
    mask = batch.mask
    y = batch.y
    n_batch = batch.graph.batch_size
    return [g, x, mask], y, n_batch, g
