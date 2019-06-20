from treeLSTM.models import TreeLSTM
from treeLSTM.dataset import TreeDataset
from treeLSTM.cells import *
import networkx as nx
import dgl
import torch.nn.functional as F
from nltk.corpus.reader import BracketParseCorpusReader
from torch.utils.data import DataLoader
from tqdm import tqdm


# TODO: modfy the dataset class according to TreeDataset
# TODO: embeddigns anf vocabulray must be loaded outside the class and sahred amogn test/train/dev
# TODO: check out to load embeddings (remove lower on emdeggings key)

class SSTDataset(TreeDataset):

    PAD_WORD = -1  # special pad word id
    UNK_WORD = -1  # out-of-vocabulary word id

    def __init__(self, path_dir, file_name_list, vocab, name):
        TreeDataset.__init__(self, path_dir, file_name_list, name)
        self.num_classes = 5
        self.max_out_degree = 2
        self.vocab = vocab
        self.__load_trees__()

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


def create_sst_model(x_size, h_size, num_classes, max_output_degree=2, dropout=0.5, pretrained_emb=None, num_vocabs=None, cell_type='nary', rank=None, pos_stationarity=False):
    if cell_type == 'nary':
        cell = NaryCell(h_size, max_output_degree, pos_stationarity=pos_stationarity)
    elif cell_type == 'hosvd':
        cell = HOSVDCell(h_size, max_output_degree, rank=rank, pos_stationarity=pos_stationarity)
    elif cell_type == 'tt':
        cell = TTCell(h_size, max_output_degree, rank=rank)
    elif cell_type == 'cancomp':
        cell = CANCOMPCell(h_size, max_output_degree, rank=rank, pos_stationarity=pos_stationarity)
    elif cell_type == 'full':
        raise ValueError('The Full Tensora agrregation cannot be used')
    else:
        raise ValueError('Cell type not known')
    if pretrained_emb is None:
        input_module = nn.Embedding(num_vocabs, x_size)
    else:
        input_module = nn.Embedding.from_pretrained(pretrained_emb, freeze=True)

    output_module = SSTOutputModule(h_size, num_classes, dropout)

    return TreeLSTM(x_size, h_size, input_module, output_module, cell)


def load_sst_dataset(vocab):
    trainset = SSTDataset('data/sst/', ['train.txt'], vocab, name='train')
    devset = SSTDataset('data/sst/', ['dev.txt'], vocab, name='dev')
    testset = SSTDataset('data/sst/', ['test.txt'], vocab, name='test')
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
