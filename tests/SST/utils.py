import torch.nn as nn
from treeLSTM import TreeLSTM, TreeDataset

import networkx as nx
import dgl
import torch.nn.functional as F
from nltk.corpus.reader import BracketParseCorpusReader
from collections import namedtuple, OrderedDict
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import torch


# TODO: modfy the dataset class according to TreeDataset
class SSTDataset(TreeDataset):

    PAD_WORD = -1  # special pad word id
    UNK_WORD = -1  # out-of-vocabulary word id

    def __init__(self, path_dir, file_name_list, name, glove300_file=None):
        TreeDataset.__init__(self, path_dir, file_name_list, name)
        self.pretrained_emb = None
        self.num_classes = 5
        # print('Preprocessing...')
        self.__load_vocabulary__()
        if glove300_file is not None:
            self.__load_embeddings__(glove300_file)
        self.__load_trees__()
        # print('Dataset creation finished. #Trees:', len(self.trees))

    def __load_vocabulary__(self):
        object_file = os.path.join(self.path_dir, 'vocab.pkl')
        text_file = os.path.join(self.path_dir, 'vocab.txt')
        if os.path.exists(object_file):
            # load vocab file
            self.vocab = torch.load(object_file)
        else:
            # create vocab file
            self.vocab = OrderedDict()
            self.logger.debug('Loading vocabulary.')
            with open(text_file, encoding='utf-8') as vf:
                for line in tqdm(vf.readlines(), desc='Loading vocabulary: '):
                    line = line.strip()
                    self.vocab[line] = len(self.vocab)
            torch.save(self.vocab, object_file)

        self.logger.info('Vocabulary loaded.')

    def __load_embeddings__(self, pretrained_emb_file):

        object_file = os.path.join(self.path_dir, 'pretrained_emb.pkl')
        if os.path.exists(object_file):
            self.pretrained_emb = torch.load(object_file)
        else:
            # filter glove
            glove_emb = {}
            self.logger.debug('Loading pretrained embeddings.')
            with open(pretrained_emb_file, 'r', encoding='utf-8') as pf:
                for line in tqdm(pf.readlines(), desc='Loading pretrained embeddings:'):
                    sp = line.split(' ')
                    if sp[0].lower() in self.vocab:
                        glove_emb[sp[0].lower()] = np.array([float(x) for x in sp[1:]])

            # initialize with glove
            pretrained_emb = []
            fail_cnt = 0
            for line in self.vocab.keys():
                if not line.lower() in glove_emb:
                    fail_cnt += 1
                pretrained_emb.append(glove_emb.get(line.lower(), np.random.uniform(-0.05, 0.05, 300)))

            self.pretrained_emb = torch.tensor(np.stack(pretrained_emb, 0))
            self.logger.info('Miss word in GloVe {0:.4f}'.format(1.0 * fail_cnt / len(self.pretrained_emb)))
            torch.save(self.pretrained_emb, object_file)

        self.logger.info('Pretrained embeddigns loaded.')

    def __load_trees__(self):
        corpus = BracketParseCorpusReader(self.path_dir, self.file_name_list)
        sents = corpus.parsed_sents(self.file_name_list)

        self.logger.debug('Loading trees.')
        # build trees
        for sent in tqdm(sents, desc='Loading trees: '):
            self.data.append(self.__build_dgl_tree__(sent))

        self.logger.info('{} trees loaded.'.format(len(self.data)))

    def get_loader(self, batch_size, device, shuffle=False):
        def batcher_dev(batch):
            batched_trees = dgl.batch(batch)
            return SSTDataset.TreeBatch(graph=batched_trees,
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
                    word = self.vocab.get(child[0].lower(), self.UNK_WORD)
                    g.add_node(cid, x=word, y=int(child.label()), mask=1)
                else:
                    g.add_node(cid, x=SSTDataset.PAD_WORD, y=int(child.label()), mask=0)
                    _rec_build(cid, child)
                g.add_edge(cid, nid)

        # add root
        g.add_node(0, x=SSTDataset.PAD_WORD, y=int(root.label()), mask=0)
        _rec_build(0, root)
        ret = dgl.DGLGraph()
        ret.from_networkx(g, node_attrs=['x', 'y', 'mask'])
        return ret

    @property
    def num_vocabs(self):
        return len(self.vocab)


class SSTOutputModule(nn.Module):

    def __init__(self, h_size, num_classes, dropout):
        super(SSTOutputModule, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(h_size, num_classes)

    def forward(self, h):
        return self.linear(self.dropout(h))


def create_sst_model(num_vocabs,
                     x_size,
                     h_size,
                     num_classes,
                     dropout,
                     pretrained_emb=None,
                     cell_type='nary', **cell_args):

    input_module = nn.Embedding(num_vocabs, x_size)
    if pretrained_emb is not None:
        input_module.weight.data.copy_(pretrained_emb)
        input_module.weight.requires_grad = True

    output_module = SSTOutputModule(h_size, num_classes, dropout)

    m = TreeLSTM(x_size, h_size, 2, input_module, output_module, cell_type, **cell_args)
    #m = TreeRNN(x_size, h_size, 2, input_module, output_module, cell_type, **cell_args)

    return m


def load_sst_dataset():
    trainset = SSTDataset('data/sst/', ['train.txt'], glove300_file='data/glove.840B.300d.txt', name='train')
    devset = SSTDataset('data/sst/', ['dev.txt'], glove300_file='data/glove.840B.300d.txt', name='dev')
    testset = SSTDataset('data/sst/', ['test.txt'], glove300_file='data/glove.840B.300d.txt', name='test')
    return trainset, devset, testset


def sst_loss_function(output_model, true_label):
    logp = F.log_softmax(output_model, 1)
    return F.nll_loss(logp, true_label, reduction='sum')


def sst_extract_batch_data(batch):
    g = batch.graph
    x = batch.x
    mask = batch.mask
    y = batch.y
    n_batch = batch.graph.batch_size
    return [g, x, mask], y, n_batch, g
