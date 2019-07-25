from treeLSTM.models import TreeLSTM
from treeLSTM.dataset import TreeDataset
from treeLSTM.cells import *
from treeLSTM.metrics import Accuracy
import networkx as nx
import dgl
import torch.nn.functional as F
from nltk import Tree
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import torch as th


class SSTDependecyDataset(TreeDataset):

    UNK_WORD = 0   # out-of-vocabulary word id
    NO_LABEL = -1  # flag to indicate no sentiment label

    def __init__(self, path_dir, file_name_list, name, vocab=None):
        TreeDataset.__init__(self, path_dir, file_name_list, name)
        self.num_classes = 5
        self.max_out_degree = -1

        if vocab is None:
            # build the vocabulary
            self.vocab = {'UNK': 0}
            self.buid_vocab = True
        else:
            self.vocab = vocab
            self.buid_vocab = False

        self.__load_sentiment_mapping__()
        self.__load_trees__()

    def __load_trees__(self):
        self.logger.debug('Loading trees.')
        # build trees
        tot_no_label = 0
        tot_nodes = 0
        for f_name in self.file_name_list:
            with open(os.path.join(self.path_dir, f_name), 'r', encoding='utf-8') as f:
                for sent in tqdm(f.readlines(), desc='Loading trees: '):
                    g, no_label = self.__build_dgl_tree__(Tree.fromstring(sent.strip('\n'), leaf_pattern='[^ ()]+', node_pattern='[^ ()]+'))
                    self.data.append(g)
                    tot_nodes += g.number_of_nodes()
                    tot_no_label += no_label

        self.logger.info('{} trees loaded.'.format(len(self.data)))
        self.logger.info('{} nodes loaded.'.format(tot_nodes))
        self.logger.info('{} nodes with no label.'.format(tot_no_label))

    def __load_sentiment_mapping__(self):
        fname_sentiment_map = 'sentiment_map.txt'

        self.phrase2sentiment = {}
        with open(os.path.join(self.path_dir, fname_sentiment_map), 'r', encoding='utf-8') as f_sentiment_map:
            for l in tqdm(f_sentiment_map.readlines(), desc='Loading sentiment labels: '):
                v = l.split('|')
                key = frozenset(v[0].split(' '))

                self.phrase2sentiment[key] = int(v[1])

    def __serach_and_update_vocabulary__(self, word):
        word = word.lower()
        idx = self.vocab.get(word, SSTDependecyDataset.UNK_WORD)

        if self.buid_vocab and idx == SSTDependecyDataset.UNK_WORD:
            self.vocab[word] = len(self.vocab)
            idx = self.vocab[word]

        return idx

    def __build_dgl_tree__(self, root):
        g = nx.DiGraph()

        phrase_no_label = []

        def _rec_build(node):
            nonlocal phrase_no_label
            phrase_subtree = []
            ch_id_list = []

            for child in node:
                if isinstance(child, str):
                    s = [child]
                    ch_id = g.number_of_nodes()
                    # add node
                    word_id = self.__serach_and_update_vocabulary__(child)
                    if frozenset(s) in self.phrase2sentiment:
                        sentiment_label = self.phrase2sentiment[frozenset(s)]
                        g.add_node(ch_id, x=word_id, y=sentiment_label, mask=1, w=child)
                    else:
                        g.add_node(ch_id, x=word_id, y=SSTDependecyDataset.NO_LABEL, mask=1, w=child)
                        phrase_no_label.append(frozenset(s))
                else:
                    s, ch_id = _rec_build(child)
                ch_id_list.append(ch_id)
                phrase_subtree += s

            phrase_subtree += [node.label()]
            word_id = self.__serach_and_update_vocabulary__(node.label())
            node_id = g.number_of_nodes()

            if frozenset(phrase_subtree) in self.phrase2sentiment:
                sentiment_label = self.phrase2sentiment[frozenset(phrase_subtree)]
                g.add_node(node_id, x=word_id, y=sentiment_label, mask=1, w=node.label())
            else:
                g.add_node(node_id, x=word_id, y=SSTDependecyDataset.NO_LABEL, mask=1, w=node.label())
                phrase_no_label.append(frozenset(phrase_subtree))

            # add edges
            assert len(ch_id_list) == len(node)
            self.max_out_degree = max(self.max_out_degree, len(node))

            for ch_id in ch_id_list:
                g.add_edge(ch_id, node_id)

            return phrase_subtree, node_id

        _rec_build(root)

        assert -1 not in[g.nodes[i]['y'] for i in range(g.number_of_nodes()) if g.out_degree(i) == 0]

        ret = dgl.DGLGraph()
        ret.from_networkx(g, node_attrs=['x', 'y', 'mask'])
        #SSTDependecyDataset.plot_tree(g)

        return ret, len(phrase_no_label)


    @staticmethod
    def plot_tree(G):
        import matplotlib.pyplot as plt
        from networkx.drawing.nx_agraph import graphviz_layout
        pos = graphviz_layout(G, prog='dot')
        nx.draw(G, pos=pos, with_labels=True, labels={u:G.nodes[u]['w'] for u in G.nodes})
        plt.axis('off')
        plt.show()


    @property
    def num_vocabs(self):
        return len(self.vocab)

    def get_vocab(self):
        return self.vocab

    def get_loader(self, batch_size, device, shuffle=False):
        def batcher_dev(batch):
            batched_trees = dgl.batch(batch)
            return SSTDependecyDataset.TreeBatch(graph=batched_trees,
                                                       mask=batched_trees.ndata['mask'].to(device),
                                                       x=batched_trees.ndata['x'].to(device),
                                                       y=batched_trees.ndata['y'].to(device))

        return DataLoader(dataset=self, batch_size=batch_size, collate_fn=batcher_dev, shuffle=shuffle,
                          num_workers=0)


def load_dataset():
    data_dir = 'data/sst_nary/dep_tree'
    train_obj_file = os.path.join(data_dir, 'train.pkl')
    val_obj_file = os.path.join(data_dir, 'val.pkl')
    test_obj_file = os.path.join(data_dir, 'test.pkl')

    if os.path.exists(train_obj_file):
        trainset = th.load(train_obj_file)
        trainset.logger.info('Training set has already parsed.')
    else:
        trainset = SSTDependecyDataset(data_dir, ['train.txt'], 'train')
        #th.save(trainset, train_obj_file)

    vocab = trainset.get_vocab()

    if os.path.exists(val_obj_file):
        valset = th.load(val_obj_file)
        valset.logger.info('Validation set has already parsed.')
    else:
        valset = SSTDependecyDataset(data_dir, ['validation.txt'], 'validation', vocab=vocab)
        #th.save(valset, val_obj_file)

    if os.path.exists(test_obj_file):
        testset = th.load(test_obj_file)
        testset.logger.info('Validation set has already parsed.')
    else:
        testset = SSTDependecyDataset(data_dir, ['test.txt'], 'test', vocab=vocab)
        #th.save(testset, test_obj_file)

    return trainset, valset, testset


def loss_function(output_model, true_label):
    idxs = (true_label != SSTDependecyDataset.NO_LABEL)
    return F.cross_entropy(output_model[idxs], true_label[idxs], reduction='sum')


class MaskedAccuracy(Accuracy):

    def update_metric(self, out, gold_label):
        pred = th.argmax(out, 1)
        idxs = (gold_label != SSTDependecyDataset.NO_LABEL)
        self.n_correct += th.sum(th.eq(gold_label[idxs], pred[idxs])).item()
        self.n_nodes += len(gold_label)


class SSTOutputModule(nn.Module):

    def __init__(self, h_size, num_classes, dropout):
        super(SSTOutputModule, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(h_size, num_classes)

    def forward(self, h):
        return self.linear(self.dropout(h))


def create_model(x_size, h_size, num_classes, max_output_degree, dropout, cell_type, rank, pos_stationarity, pretrained_emb=None, num_vocabs=None):
    if cell_type == 'nary':
        cell = NaryCell(h_size, max_output_degree, pos_stationarity=pos_stationarity)
    elif cell_type == 'hosvd':
        cell = HOSVDCell(h_size, max_output_degree, rank=rank, pos_stationarity=pos_stationarity)
    elif cell_type == 'tt':
        cell = TTCell(h_size, max_output_degree, rank=rank, pos_stationarity=pos_stationarity)
    elif cell_type == 'cancomp':
        cell = CANCOMPCell(h_size, max_output_degree, rank=rank, pos_stationarity=pos_stationarity)
    elif cell_type == 'full':
        cell = BinaryFullTensorCell(h_size, max_output_degree, pos_stationarity=pos_stationarity)
    else:
        raise ValueError('Cell type not known')
    if pretrained_emb is None:
        input_module = nn.Embedding(num_vocabs, x_size)
    else:
        input_module = nn.Embedding.from_pretrained(pretrained_emb, freeze=False)

    output_module = SSTOutputModule(h_size, num_classes, dropout)

    return TreeLSTM(x_size, h_size, input_module, output_module, cell)


def extract_batch_data(batch):
    g = batch.graph
    x = batch.x
    mask = batch.mask
    y = batch.y
    return [g, x, mask], y, g
