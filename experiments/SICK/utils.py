import networkx as nx
import dgl
import torch.nn.functional as F
from nltk.corpus.reader import Tree
from collections import namedtuple
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

from treeLSTM.__old__.cells_old import *
from treeLSTM.models import TreeLSTM
from treeLSTM.dataset import TreeDataset
from treeLSTM.metrics import MSE, Pearson
from treeLSTM.trainer import train_and_validate, test

from experiments.execution_utils import load_embeddings, init_base_logger, get_base_logger, get_sub_logger

import pickle
import torch.nn.init as INIT
import torch.optim as optim


class SICKDataset(TreeDataset):

    PAD_WORD = -1  # special pad word id
    UNK_WORD = 0  # out-of-vocabulary word id

    SICKBatch = namedtuple('SICKBatch', ['batch_a', 'batch_b', 'score', 'target_distr'])

    NUM_CLASSES = 5

    def __init__(self, path_dir, file_name_list, name, vocab=None):
        TreeDataset.__init__(self, path_dir, file_name_list, name)
        self.max_out_degree = -1

        if vocab is None:
            # build the vocabulary
            self.vocab = {'UNK': 0}
            self.build_vocab = True
        else:
            self.vocab = vocab
            self.build_vocab = False

        self.__load_trees__()

    def __load_trees__(self):
        A_file_list = list(map(lambda x: x.replace('.txt', '_A.txt'), self.file_name_list))
        B_file_list = list(map(lambda x: x.replace('.txt', '_B.txt'), self.file_name_list))
        SCORE_file_list = list(map(lambda x: x.replace('.txt', '_SCORE.txt'), self.file_name_list))
        self.logger.debug('Loading trees.')

        tree_a = []
        with open(os.path.join(self.path_dir, A_file_list[0]), 'r') as f_A:
            for l in tqdm(f_A.readlines(), desc='Loading A trees: '):
                tree_a.append(Tree.fromstring(l))

        tree_b = []
        with open(os.path.join(self.path_dir, B_file_list[0]), 'r') as f_B:
            for l in tqdm(f_B.readlines(), desc='Loading B trees: '):
                tree_b.append(Tree.fromstring(l))

        with open(os.path.join(self.path_dir, SCORE_file_list[0]), 'r') as f_SCORE:
            for i,l in tqdm(enumerate(f_SCORE.readlines()), total=len(tree_a), desc='Parsing trees: '):
                s = float(l[:-1])
                a_t = self.__build_dgl_tree__(tree_a[i])
                b_t = self.__build_dgl_tree__(tree_b[i])
                self.data.append((s, a_t, b_t))

        self.logger.info('{} trees loaded.'.format(len(self.data)))

    def __serach_and_update_vocabulary__(self, word):
        word = word.lower()
        idx = self.vocab.get(word, SICKDataset.UNK_WORD)

        if self.build_vocab and idx == SICKDataset.UNK_WORD:
            self.vocab[word] = len(self.vocab)
            idx = self.vocab[word]

        return idx

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
                    w_id = self.__serach_and_update_vocabulary__(child)
                    assert w_id != -1
                    g.add_node(cid, x=w_id, y=-1, mask=1)
                else:
                    # internal node
                    w_id = self.__serach_and_update_vocabulary__(child.label())
                    assert w_id != -1
                    g.add_node(cid, x=w_id, y=-1, mask=1)
                    _rec_build(cid, child)
                g.add_edge(cid, nid)

        # add root
        w_id = self.__serach_and_update_vocabulary__(root.label())
        assert w_id != -1
        g.add_node(0, x=w_id, y=-1, mask=1)
        _rec_build(0, root)
        ret = dgl.DGLGraph()
        ret.from_networkx(g, node_attrs=['x', 'y', 'mask'])
        return ret

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

    @property
    def num_vocabs(self):
        return len(self.vocab)

    def get_vocab(self):
        return self.vocab


def load_SICK_dataset():
    data_dir = 'data/sick/'
    train_obj_file = os.path.join(data_dir, 'train.pkl')
    val_obj_file = os.path.join(data_dir, 'val.pkl')
    test_obj_file = os.path.join(data_dir, 'test.pkl')

    if os.path.exists(train_obj_file):
        with open(train_obj_file, 'rb') as f:
            trainset = pickle.load(f)
        get_sub_logger('train').info('Training set has already parsed.')
    else:
        trainset = SICKDataset(data_dir, ['SICK_train.txt'], 'train')
        with open(train_obj_file, 'wb') as f:
            pickle.dump(trainset, f)

    vocab = trainset.get_vocab()

    if os.path.exists(val_obj_file):
        with open(val_obj_file, 'rb') as f:
            valset = pickle.load(f)
        get_sub_logger('validation').info('Validation set has already parsed.')
    else:
        valset = SICKDataset(data_dir, ['SICK_trial.txt'], 'validation', vocab=vocab)
        with open(val_obj_file, 'wb') as f:
            pickle.dump(valset, f)

    if os.path.exists(test_obj_file):
        with open(test_obj_file, 'rb') as f:
            testset = pickle.load(f)
        get_sub_logger('test').info('Test set has already parsed.')
    else:
        testset = SICKDataset(data_dir, ['SICK_test.txt'], 'test', vocab=vocab)
        with open(test_obj_file, 'wb') as f:
            pickle.dump(testset, f)

    return trainset, valset, testset


class SICKComparisonModule(nn.Module):

    def __init__(self, input_dim, h_dim, num_classes):
        super(SICKComparisonModule, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = h_dim
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

    def __init__(self, x_size, h_size, comb_h_dim, num_classes, cell, pretrained_emb=None, num_vocabs=None):
        super(SICKModel, self).__init__()

        if pretrained_emb is None:
            input_module = nn.Embedding(num_vocabs, x_size)
        else:
            input_module = nn.Embedding.from_pretrained(pretrained_emb, freeze=True)

        output_module = nn.Identity()

        self.tree_model = TreeLSTM(x_size, h_size, input_module, output_module, cell)
        self.comb_module = SICKComparisonModule(h_size, comb_h_dim, num_classes)

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


def create_SICK_model(x_size, h_size, comb_h_dim, num_classes, max_output_degree, cell_type, rank, pos_stationarity, scale_factor, pretrained_emb=None, num_vocabs=None):
    if cell_type == 'sumchild':
        cell = SumChildCell(h_size, max_output_degree, pos_stationarity=pos_stationarity)
    elif cell_type == 'hosvd':
        cell = HOSVDCell(h_size, max_output_degree, rank=rank, pos_stationarity=pos_stationarity)
    elif cell_type == 'tt':
        cell = TTCell(h_size, max_output_degree, rank=rank, pos_stationarity=pos_stationarity)
    elif cell_type == 'cancomp':
        cell = CANCOMPCell(h_size, max_output_degree, rank=rank, pos_stationarity=pos_stationarity, scale_factor=scale_factor)
    elif cell_type == 'cancomp_input':
        cell = CANCOMPCellInput(h_size, max_output_degree, rank=rank, pos_stationarity=pos_stationarity)
    elif cell_type == 'full':
        cell = BinaryFullTensorCell(h_size, max_output_degree, pos_stationarity=pos_stationarity)
    elif cell_type == 'sumchild_input':
        cell = SumChildALLInputDependentCell(h_size, max_output_degree, pos_stationarity=pos_stationarity)
    elif cell_type == 'sumchild_f_input':
        cell = SumChildFInputDependentCell(h_size, max_output_degree, pos_stationarity=pos_stationarity)
    elif cell_type == 'sumchild_iou_input':
        cell = SumChildIOUInputDependentCell(h_size, max_output_degree, pos_stationarity=pos_stationarity)
    else:
        raise ValueError('Cell type not known')

    return SICKModel(x_size, h_size, comb_h_dim, num_classes, cell, pretrained_emb, num_vocabs)


def SICK_loss_function(output_model, true_label):
    return F.kl_div(output_model[0], true_label[0], reduction='batchmean')


def extract_SICK_batch_data(batch):
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

    return [[g_a, x_a, mask_a], [g_b, x_b, mask_b]], [target_distr, sym], None


class MseSICK(MSE):

    def update_metric(self, out, gold_label):
        super(MseSICK, self).update_metric(out[1], gold_label[1])


class PearsonSICK(Pearson):

    def update_metric(self, out, gold_label):
        super(PearsonSICK, self).update_metric(out[1], gold_label[1])


def SICK_single_run_fun(args, device, log_dir):
    # load the data
    trainset, devset, testset = load_SICK_dataset()

    pretrained_embs = load_embeddings('data/sick/', pretrained_emb_file='data/glove.840B.300d.txt', vocab=trainset.get_vocab())

    # create the model
    model = create_SICK_model(args.x_size, args.h_size, args.comb_h_size, trainset.NUM_CLASSES,
                              max_output_degree=trainset.max_out_degree,
                              pretrained_emb=pretrained_embs,
                              cell_type=args.cell_type,
                              rank=args.rank,
                              pos_stationarity=args.pos_stationarity,
                              scale_factor=args.scale_factor).to(device)

    # create the optimizser
    params_no_cell = [x[1] for x in list(model.named_parameters()) if
                      x[1].requires_grad and x[1].size(0) != trainset.num_vocabs and 'cell' not in x[0]]
    params_cell = [x[1] for x in list(model.named_parameters()) if
                   x[1].requires_grad and x[1].size(0) != trainset.num_vocabs and 'cell' in x[0]]

    for p in params_cell:
        if p.dim() > 1:
            INIT.xavier_uniform_(p)

    for p in params_no_cell:
        if p.dim() > 1:
            INIT.xavier_uniform_(p)

    # create the optimizer
    optimizer = optim.Adagrad([
        {'params': params_no_cell, 'lr': args.lr, 'weight_decay': args.weight_decay},
        {'params': params_cell, 'lr': args.lr, 'weight_decay': args.cell_weight_decay}])

    # train and validate
    best_model, info_training = train_and_validate(model, extract_SICK_batch_data, SICK_loss_function, optimizer, trainset,
                                                   devset, device,
                                                   metrics_class=[MseSICK, PearsonSICK],
                                                   batch_size=args.batch_size,
                                                   n_epochs=args.epochs, early_stopping_patience=args.early_stopping,
                                                   evaluate_on_training_set=True)

    th.save(best_model, os.path.join(log_dir, 'best_model.pkl'))
    th.save(info_training, os.path.join(log_dir, 'info_training.pkl'))

    # test
    test_metrics, test_predictions = test(best_model, extract_SICK_batch_data, testset, device,
                                          metrics_class=[MseSICK, PearsonSICK],
                                          batch_size=args.batch_size)

    th.save({'test_metrics': test_metrics, 'test_predictions': test_predictions},
            os.path.join(log_dir, 'test_presults.pkl'))


def get_SICK_model_selection_fun(args, device):

    def train_foo(id, log_dir, params):
        init_base_logger(log_dir, 'exp{}'.format(id))
        logger = get_base_logger()
        logger.info(str(params))

        # load the data
        trainset, devset, testset = load_SICK_dataset()

        pretrained_embs = load_embeddings('data/sick/', pretrained_emb_file='data/glove.840B.300d.txt',
                                          vocab=trainset.get_vocab())

        # create the model
        model = create_SICK_model(params['x_size'], params['h_size'],params['comb_h_size'], trainset.NUM_CLASSES,
                                  max_output_degree=trainset.max_out_degree,
                                  pretrained_emb=pretrained_embs,
                                  cell_type=args.cell_type,
                                  rank=params['rank'],
                                  pos_stationarity=params['pos_stationarity'],
                                  scale_factor=params['scale_factor']).to(device)

        # create the optimizser
        params_no_cell = [x[1] for x in list(model.named_parameters()) if
                          x[1].requires_grad and x[1].size(0) != trainset.num_vocabs and 'cell' not in x[0]]
        params_cell = [x[1] for x in list(model.named_parameters()) if
                       x[1].requires_grad and x[1].size(0) != trainset.num_vocabs and 'cell' in x[0]]

        for p in params_cell:
            if p.dim() > 1:
                INIT.xavier_uniform_(p)

        for p in params_no_cell:
            if p.dim() > 1:
                INIT.xavier_uniform_(p)

        # create the optimizer
        optimizer = optim.Adagrad([
            {'params': params_no_cell, 'lr': params['lr'], 'weight_decay': params['weight_decay']},
            {'params': params_cell, 'lr': params['lr'], 'weight_decay': params['cell_weight_decay']}])

        # train and validate
        best_model, info_training = train_and_validate(model, extract_SICK_batch_data, SICK_loss_function, optimizer,
                                                       trainset,
                                                       devset, device,
                                                       metrics_class=[MseSICK, PearsonSICK],
                                                       batch_size=args.batch_size,
                                                       n_epochs=args.epochs,
                                                       early_stopping_patience=args.early_stopping,
                                                       evaluate_on_training_set=True)

        th.save(best_model, os.path.join(log_dir, 'best_model.pkl'))
        th.save(info_training, os.path.join(log_dir, 'info_training.pkl'))

        ris = {}
        best_epoch = info_training['best_epoch']
        ris['MSE_tr'] = info_training['tr_metrics'][MseSICK.__name__][best_epoch]
        ris['Pearson_tr'] = info_training['tr_metrics'][PearsonSICK.__name__][best_epoch]
        ris['MSE_val'] = info_training['dev_metrics'][MseSICK.__name__][best_epoch]
        ris['Pearson_val'] = info_training['dev_metrics'][PearsonSICK.__name__][best_epoch]
        return ris

    return train_foo
