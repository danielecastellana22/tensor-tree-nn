import torch.nn as nn
from treeLSTM import TreeLSTM, TreeDataset

import networkx as nx
import dgl
from collections import namedtuple
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
from nltk import Tree


class ToyDataset(TreeDataset):

    ToyBatch = namedtuple('XORBatch', ['graph', 'mask', 'x', 'label'])

    def __init__(self, path_dir, file_name):
        TreeDataset.__init__(self, path_dir, file_name)
        self.__load_trees__()

    def __load_trees__(self):
        self.logger.debug('Loading trees.'.format(len(self.trees)))
        # build trees
        with open(os.path.join(self.path_dir, self.file_name),'r') as txtfile:
            for sent in tqdm(txtfile.readlines(), desc='Loading trees: '):
                t = Tree.fromstring(sent)
                self.trees.append(self.__build_tree__(t))

        self.logger.info('{} trees loaded.'.format(len(self.trees)))

    def __build_tree__(self, root):
        g = nx.DiGraph()

        def _rec_build(nid, node):
            for child in node:
                cid = g.number_of_nodes()
                if isinstance(child[0], str) or isinstance(child[0], bytes):
                    # leaf node
                    word = 0 if child[0].lower() == 'a' else 1
                    g.add_node(cid, x=word, y=int(child.label()), mask=1)
                else:
                    g.add_node(cid, x=-1    , y=int(child.label()), mask=0)
                    _rec_build(cid, child)
                g.add_edge(cid, nid)

        # add root
        g.add_node(0, x=-1, y=int(root.label()), mask=0)
        _rec_build(0, root)
        ret = dgl.DGLGraph()
        ret.from_networkx(g, node_attrs=['x', 'y', 'mask'])
        return ret

    def get_loader(self, batch_size, device, shuffle=False):
        def batcher_dev(batch):
            batch_trees = dgl.batch(batch)
            return ToyDataset.ToyBatch(graph=batch_trees,
                                       mask=batch_trees.ndata['mask'].to(device),
                                       x=batch_trees.ndata['x'].to(device),
                                       label=batch_trees.ndata['y'].to(device))

        return DataLoader(dataset=self, batch_size=batch_size, collate_fn=batcher_dev, shuffle=shuffle,
                          num_workers=0)


class HTENSOutputModule(nn.Module):

    def __init__(self,h_size, num_classes, dropout):
        super(HTENSOutputModule, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(h_size, num_classes)

    def forward(self, h):
        return self.linear(self.dropout(h))


def create_htens_model(x_size,
                     h_size,
                     dropout,
                     cell_type='nary', **cell_args):

    num_classes = 2
    num_vocabs = 2
    input_module = nn.Embedding(num_vocabs, x_size)

    output_module = HTENSOutputModule(h_size, num_classes, dropout)

    m = TreeLSTM(x_size, h_size, 2, input_module, output_module, cell_type, **cell_args)

    return m


def load_htens_dataset():
    trainset = ToyDataset('data/htens/', 'train.txt')
    devset = ToyDataset('data/htens/', 'dev.txt')
    testset = ToyDataset('data/htens/', 'test.txt')

    return trainset, devset, testset

