from abc import ABC, abstractmethod
import networkx as nx
import dgl
import dgl.backend as F
from nltk.corpus.reader import BracketParseCorpusReader
from nltk import Tree
from collections import namedtuple, OrderedDict
import numpy as np
from torch.utils.data import DataLoader
from .utils import get_new_logger
from tqdm import tqdm
import os
import torch

# TODO: use nltk/Tree

class TreeDataset(ABC):

    def __init__(self, path_dir, file_name):
        self.trees = []
        self.path_dir = path_dir
        self.file_name = file_name

        self.logger = get_new_logger('loading.{}'.format(os.path.join(path_dir,file_name)))

    def __getitem__(self, idx):
        return self.trees[idx]

    def __len__(self):
        return len(self.trees)

    @abstractmethod
    def __load_trees__(self):
        raise NotImplementedError('users must define __load__ to use this base class')

    @abstractmethod
    def get_loader(self, batch_size, device, shuffle=False):
        raise NotImplementedError('users must define __load__ to use this base class')


class SSTDataset(TreeDataset):
        """Stanford Sentiment Treebank dataset.

        Each sample is the constituency tree of a sentence. The leaf nodes
        represent words. The word is a int value stored in the ``x`` feature field.
        The non-leaf node has a special value ``PAD_WORD`` in the ``x`` field.
        Each node also has a sentiment annotation: 5 classes (very negative,
        negative, neutral, positive and very positive). The sentiment label is a
        int value stored in the ``y`` feature field.

        .. note::
            This dataset class is compatible with pytorch's :class:`Dataset` class.

        .. note::
            All the samples will be loaded and preprocessed in the memory first.

        Parameters
        ----------
        mode : str, optional
            Can be ``'train'``, ``'val'``, ``'test'`` and specifies which data file to use.
        vocab_file : str, optional
            Optional vocabulary file.
        """
        PAD_WORD = -1  # special pad word id
        UNK_WORD = -1  # out-of-vocabulary word id

        SSTBatch = namedtuple('SSTBatch', ['graph', 'mask', 'x', 'label'])

        def __init__(self, path_dir, file_name, glove300_file=None):
            TreeDataset.__init__(self, path_dir, file_name)
            self.pretrained_emb = None
            self.num_classes = 5
            #print('Preprocessing...')
            self.__load_vocabulary__()
            if glove300_file is not None:
                self.__load_embeddings__(glove300_file)
            self.__load_trees__()
            #print('Dataset creation finished. #Trees:', len(self.trees))

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

                self.pretrained_emb = F.tensor(np.stack(pretrained_emb, 0))
                self.logger.info('Miss word in GloVe {0:.4f}'.format(1.0 * fail_cnt / len(self.pretrained_emb)))
                torch.save(self.pretrained_emb, object_file)

            self.logger.info('Pretrained embeddigns loaded.')

        def __load_trees__(self):
            corpus = BracketParseCorpusReader(self.path_dir, [self.file_name])
            sents = corpus.parsed_sents(self.file_name)

            self.logger.debug('Loading trees.'.format(len(self.trees)))
            # build trees
            for sent in tqdm(sents, desc='Loading trees: '):
                self.trees.append(self.__build_tree__(sent))

            self.logger.info('{} trees loaded.'.format(len(self.trees)))

        def __build_tree__(self, root):
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

        def get_loader(self, batch_size, device, shuffle=False):
            def batcher_dev(batch):
                batch_trees = dgl.batch(batch)
                return SSTDataset.SSTBatch(graph=batch_trees,
                                mask=batch_trees.ndata['mask'].to(device),
                                x=batch_trees.ndata['x'].to(device),
                                label=batch_trees.ndata['y'].to(device))

            return DataLoader(dataset=self, batch_size=batch_size, collate_fn=batcher_dev, shuffle=shuffle,
                                         num_workers=0)

        @property
        def num_vocabs(self):
            return len(self.vocab)


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
