import torch.nn.functional as F
import pickle

from nltk import Tree

from treeRNN.aggregators import TypedAggregator
from treeRNN.dataset import TreeDataset
from treeRNN.metrics import Accuracy
from treeRNN.trainer import *

from experiments.execution_utils import init_base_logger, get_base_logger, get_tree_model_class, get_aggregator_class
import networkx as nx
import dgl
from collections import namedtuple
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.init as INIT
import torch.optim as optim
from tqdm import tqdm
import os


def binarize_tree(node: Tree):
    if isinstance(node, str):
        # is a leaf
        return node
    else:
        # is internal
        ch_list = []
        for ch in node:
            ch_list.append(binarize_tree(ch))

        opNode = node.label()
        newNode = Tree('', [opNode, ch_list[0]])
        for newCh in ch_list[1:]:
            newNode = Tree('', [newNode, newCh])
        newNode = Tree('', [newNode, ']'])
        return newNode


class ListOpsDatasetTyped(TreeDataset):

    ListOpsBatch = namedtuple('ListOpsBatch', ['batch_a', 'result'])

    NUM_CLASSES = 10 # the digit
    NUM_INPUTS = 10 # the digits

    def __init__(self, path_dir, file_name_list, name, binary=False, input_vocabolary=None, output_vocabulary=None, operator_vocabulary=None):
        TreeDataset.__init__(self, path_dir, file_name_list, name)

        self.max_output_degree = 0

        if operator_vocabulary is not None:
            self.operator_vocabulary = operator_vocabulary
        else:
            self.operator_vocabulary = {}

        if input_vocabolary is not None:
            self.input_vocabulary = input_vocabolary
        else:
            self.__create_input_vocabulary__()

        if output_vocabulary is not None:
            self.output_vocabulary = output_vocabulary
        else:
            self.__create_output_vocabulary__()

        self.__load_trees__(binary)
        self.logger.info("{} input words, max-out-degree of {}.".format(self.get_num_vocabs(), self.max_output_degree))

    def __create_input_vocabulary__(self):
        self.input_vocabulary = {}
        # add digit from 0 to 9
        for i in range(self.NUM_INPUTS):
            ch = chr(ord('0') + i)
            self.input_vocabulary[ch] = len(self.input_vocabulary)

    def __create_output_vocabulary__(self):
        self.output_vocabulary = {}
        rev_output_vocabulary = [chr(ord('0') + i) for i in range(self.NUM_CLASSES)]
        for s in rev_output_vocabulary:
            self.output_vocabulary[s] = len(self.output_vocabulary)

    def __load_trees__(self, binary):
        self.logger.debug('Loading trees.')
        # build trees
        f_names = [os.path.join(self.path_dir,x) for x in self.file_name_list]

        for f in f_names:
            with open(f, 'r') as txtfile:
                for sent in tqdm(txtfile.readlines(), desc='Loading trees: '):
                    l_sent = sent[:-1].split('\t')
                    out = self.output_vocabulary[l_sent[0]]

                    if l_sent[1][0] == '(':
                        a_t = Tree.fromstring(l_sent[1])
                    else:
                        # one node
                        a_t = l_sent[1]

                    a_t = self.__build_dgl_tree__(a_t, binary)
                    self.data.append((out, a_t))

        self.logger.info('{} trees loaded.'.format(len(self.data)))

    def __get_id_operator__(self, op):
        if op not in self.operator_vocabulary:
            self.operator_vocabulary[op] = len(self.operator_vocabulary)
        return self.operator_vocabulary[op]

    def __build_dgl_tree__(self, root, binary):
        g = nx.DiGraph()

        def _rec_build(nid, node):

            if isinstance(node, str):
                # is a leaf
                if node in self.input_vocabulary:
                    w = self.input_vocabulary[node]
                else:
                    # in binary tree the operators are on leaves
                    w = len(self.input_vocabulary)
                    self.input_vocabulary[node] = w
                g.add_node(nid, type=-1, x=w, y=-1, mask=1)
            else:
                self.max_output_degree = max(self.max_output_degree, len(node))
                # is internal
                if binary:
                    g.add_node(nid, type=-1, x=-1, y=-1, mask=0)
                else:
                    g.add_node(nid, type=self.__get_id_operator__(node.label()), x=-1, y=-1, mask=0)

                for ch in node:
                    cid = g.number_of_nodes()
                    _rec_build(cid, ch)
                    g.add_edge(cid, nid)

        if binary:
            root = binarize_tree(root)
        _rec_build(0, root)
        ret = dgl.DGLGraph()
        ret.from_networkx(g, node_attrs=['x', 'y', 'mask', 'type'])

        return ret

    def get_loader(self, batch_size, device, shuffle=False):
        def batcher_dev(tuple_data):
            #tuple data contains (symbol,ltree,rtree)
            results_list, graph_a_list = zip(*tuple_data)
            batched_a_trees = dgl.batch(graph_a_list)
            tupled_batch_a = ListOpsDatasetTyped.TreeBatch(graph=batched_a_trees,
                                                           mask=batched_a_trees.ndata['mask'].to(device),
                                                           x=batched_a_trees.ndata['x'].to(device),
                                                           y=batched_a_trees.ndata['y'].to(device))

            return ListOpsDatasetTyped.ListOpsBatch(batch_a=tupled_batch_a,
                                                    result=th.LongTensor(results_list).to(device))

        return DataLoader(dataset=self, batch_size=batch_size, collate_fn=batcher_dev, shuffle=shuffle,
                          num_workers=0)

    def get_num_operators(self):
        return len(self.operator_vocabulary)

    def get_num_vocabs(self):
        return len(self.input_vocabulary)


def load_ListOps_dataset_typed(data_dir, binary=False, load_test=True):
    data_dir = os.path.join('data', data_dir)
    if binary:
        train_obj_file = os.path.join(data_dir, 'train_bin_cl.pkl')
        dev_obj_file = os.path.join(data_dir, 'val_bin_cl.pkl')
        test_obj_file = os.path.join(data_dir, 'test_bin_cl.pkl')
    else:
        train_obj_file = os.path.join(data_dir, 'train_typed.pkl')
        dev_obj_file = os.path.join(data_dir, 'val_typed.pkl')
        test_obj_file = os.path.join(data_dir, 'test_typed.pkl')

    # training set
    if os.path.exists(train_obj_file):
        with open(train_obj_file, 'rb') as f:
            trainset = pickle.load(f)
        get_sub_logger('train').info('Training set has already parsed.')
    else:
        trainset = ListOpsDatasetTyped(data_dir, ['train.txt'], name='train_set', binary=binary)
        with open(train_obj_file, 'wb') as f:
            pickle.dump(trainset, f)

    # validation set
    if os.path.exists(dev_obj_file):
        with open(dev_obj_file, 'rb') as f:
            devset = pickle.load(f)
        get_sub_logger('validation').info('Validation set has already parsed.')
    else:
        devset = ListOpsDatasetTyped(data_dir, ['dev.txt'], name='dev_set', binary=binary,
                                     input_vocabolary=trainset.input_vocabulary,
                                     output_vocabulary=trainset.output_vocabulary,
                                     operator_vocabulary=trainset.operator_vocabulary)
        with open(dev_obj_file, 'wb') as f:
            pickle.dump(devset, f)

    # test set
    testset = None
    if load_test:
        if os.path.exists(test_obj_file):
            with open(test_obj_file, 'rb') as f:
                testset = pickle.load(f)
            get_sub_logger('test').info('Test set has already parsed.')
        else:
            testset = ListOpsDatasetTyped(data_dir, ['test.txt'], name='test_set', binary=binary,
                                          input_vocabolary=trainset.input_vocabulary,
                                          output_vocabulary=trainset.output_vocabulary,
                                          operator_vocabulary=trainset.operator_vocabulary)
            with open(test_obj_file, 'wb') as f:
                pickle.dump(testset, f)

    return trainset, devset, testset


def ListOps_loss_function(output_model, true_label):
    return F.cross_entropy(output_model, true_label, reduction='sum')


def extract_ListOps_batch_data(batch):
    a_batched = batch.batch_a
    ris = batch.result

    g_a = a_batched.graph
    x_a = a_batched.x
    mask_a = a_batched.mask

    return [[g_a, x_a, mask_a]], ris, None


class ListOpsComparisonModule(nn.Module):

    def __init__(self, in_size, h_size, out_size):
        super(ListOpsComparisonModule, self).__init__()
        self.l1 = nn.Linear(in_size, h_size)
        self.l2 = nn.Linear(h_size, h_size)
        self.l3 = nn.Linear(h_size, out_size)

    # neighbour_states has shape batch_size x n_neighbours x insize
    def forward(self, h_root):
        return self.l3(F.relu(self.l2(F.relu(self.l1(h_root)))))


class ListOpsModel(nn.Module):

    def __init__(self,use_one_hot_encoding, max_out_degree, num_vocabs, mlp_h_size,
                 tree_model_class, x_size, h_size, pos_stationarity,
                aggregator_class, **kwargs):
        super(ListOpsModel, self).__init__()

        if not use_one_hot_encoding:
            input_module = nn.Embedding(num_vocabs, x_size)
        else:
            #input_module = nn.Embedding.from_pretrained(th.eye(num_vocabs), freeze=True)
            input_module = nn.Embedding.from_pretrained(th.tril(th.ones((num_vocabs, num_vocabs))), freeze=True)
        output_module = nn.Identity()

        self.tree_model = tree_model_class(x_size, h_size, max_out_degree, pos_stationarity, input_module, output_module,
                                           aggregator_class, **kwargs)

        self.comb_module = ListOpsComparisonModule(h_size, mlp_h_size, ListOpsDatasetTyped.NUM_CLASSES)

    def forward(self, data_a):
        h_a_tree = self.tree_model(*data_a)

        g_a = data_a[0]

        root_id_a = [i for i in range(g_a.number_of_nodes()) if g_a.out_degree(i) == 0]

        h_root_a = h_a_tree[root_id_a]

        return self.comb_module(h_root_a)


def create_ListOps_model(use_one_hot_encoding, typed, max_out_degree, num_operators, num_vocabs, mlp_h_size,
                     tree_model_type, x_size, h_size, pos_stationarity,
                     cell_type, rank):

    tree_model_class = get_tree_model_class(tree_model_type)
    agg_class = get_aggregator_class(cell_type)

    if typed:
        return ListOpsModel(use_one_hot_encoding, max_out_degree, num_vocabs, mlp_h_size,
                            tree_model_class, x_size, h_size, pos_stationarity,
                            TypedAggregator, n_type=num_operators, agg_class=agg_class, rank=rank)
    else:
        return ListOpsModel(use_one_hot_encoding, max_out_degree, num_vocabs, mlp_h_size,
                            tree_model_class, x_size, h_size, pos_stationarity,
                            agg_class, rank=rank)


def ListOps_single_run_fun(args, device, log_dir):

    if args.typed_data and args.binary_tree:
        raise ValueError('You can specify either typed or binary data')

    if not args.typed_data and not args.binary_tree:
        raise ValueError('You must specify either typed or binary data')

    if not hasattr(args, 'use_one_hot'):
        args.use_one_hot = False

    if args.typed_data:
        # load the data
        trainset, devset, testset = load_ListOps_dataset_typed(args.dataset, binary=False)
    else:
        # load the data
        trainset, devset, testset = load_ListOps_dataset_typed(args.dataset, binary=args.binary_tree)

    num_vocabs = trainset.get_num_vocabs()

    if args.use_one_hot:
        get_base_logger().info('x_size set to {} due to the one-hot encoding.'.format(num_vocabs))
        args.x_size = num_vocabs

    # create the model
    model = create_ListOps_model(args.use_one_hot, args.typed_data,
                                 trainset.max_output_degree, trainset.get_num_operators(), num_vocabs, args.mlp_h_size,
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
    # optimizer = optim.Adagrad([
    #    {'params': params_no_cell, 'lr': args.lr, 'weight_decay': args.weight_decay},
    #    {'params': params_cell, 'lr': args.lr, 'weight_decay': args.weight_decay}])

    optimizer = optim.Adadelta([
        {'params': params_no_cell, 'weight_decay': args.weight_decay},
        {'params': params_cell, 'weight_decay': args.weight_decay}])

    # train and validate
    best_model, info_training = train_and_validate(model, extract_ListOps_batch_data, ListOps_loss_function, optimizer, trainset,
                                                   devset, device,
                                                   metrics_class=[Accuracy],
                                                   batch_size=args.batch_size,
                                                   n_epochs=args.epochs, early_stopping_patience=args.early_stopping,
                                                   evaluate_on_training_set=True)

    th.save(best_model, os.path.join(log_dir, 'best_model.pkl'))
    th.save(info_training, os.path.join(log_dir, 'info_training.pkl'))

    # test
    tm, tp = test(best_model, extract_ListOps_batch_data, testset, device,
                  metrics_class=[Accuracy],
                  batch_size=args.batch_size)

    th.save({'test_metrics': tm, 'test_predictions': tp},
            os.path.join(log_dir, 'test_results.pkl'))


def get_ListOps_model_selection_fun(args, device):

    if args.typed_data and args.binary_tree:
        raise ValueError('You can specify either typed or binary data')

    if not args.typed_data and not args.binary_tree:
        raise ValueError('You must specify either typed or binary data')

    if not hasattr(args, 'use_one_hot'):
        args.use_one_hot = False

    def train_foo(id, log_dir, params):
        init_base_logger(log_dir, 'exp{}'.format(id))
        logger = get_base_logger()
        logger.info(str(params))

        if args.typed_data:
            # load the data
            trainset, devset, _ = load_ListOps_dataset_typed(args.dataset, binary=False, load_test=False)
        else:
            # load the data
            trainset, devset, _ = load_ListOps_dataset_typed(args.dataset, binary=args.binary_tree, load_test=False)

        num_vocabs = trainset.get_num_vocabs()

        if args.use_one_hot:
            get_base_logger().info('x_size set to {} due to the one-hot encoding.'.format(num_vocabs))
            args.x_size = num_vocabs

            # create the model with word-emveddings
            model = create_ListOps_model(args.use_one_hot, args.typed_data,
                                         trainset.max_output_degree, trainset.get_num_operators(), num_vocabs, args.mlp_h_size,
                                         args.tree_model, args.x_size, params['h_size'], args.pos_stationarity,
                                         args.cell_type, rank=params['rank']).to(device)
        else:
            # create the model with word-emveddings
            model = create_ListOps_model(args.use_one_hot, args.typed_data,
                                         trainset.max_output_degree, trainset.get_num_operators(), num_vocabs, args.mlp_h_size,
                                         args.tree_model, params['x_size'], params['h_size'], args.pos_stationarity,
                                         args.cell_type, rank=params['rank']).to(device)

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
            {'params': params_no_cell, 'weight_decay': params['weight_decay']},
            {'params': params_cell, 'weight_decay': params['weight_decay']}])

        # train and validate
        best_model, info_training = train_and_validate(model, extract_ListOps_batch_data, ListOps_loss_function, optimizer, trainset,
                                                       devset, device,
                                                       metrics_class=[Accuracy],
                                                       batch_size=args.batch_size,
                                                       n_epochs=args.epochs, early_stopping_patience=args.early_stopping,
                                                       evaluate_on_training_set=True)

        th.save(best_model, os.path.join(log_dir, 'best_model.pkl'))
        th.save(info_training, os.path.join(log_dir, 'info_training.pkl'))

        ris = {}
        best_epoch = info_training['best_epoch']
        ris['tr_acc'] = info_training['tr_metrics'][Accuracy.__name__][best_epoch]
        ris['dev_acc'] = info_training['dev_metrics'][Accuracy.__name__][best_epoch]

        return ris

    return train_foo