import torch.nn.functional as F

from treeRNN.dataset import TreeDataset
from treeRNN.metrics import Accuracy, RootAccuracy, LeavesAccuracy
from treeRNN.trainer import *
from treeRNN.aggregators import BaseAggregator

from experiments.execution_utils import init_base_logger, get_base_logger, get_aggregator_class, get_tree_model_class
import networkx as nx
import dgl
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.init as INIT
import torch.optim as optim
from tqdm import tqdm
import os
from nltk import Tree


class ToyDataset(TreeDataset):

    def __init__(self, path_dir, file_name_list, name, n_in_vocab=2, n_out_vocab=2, operator_vocabulary=None):
        TreeDataset.__init__(self, path_dir, file_name_list, name)
        self.max_out_degree = 0
        self.num_vocabs = n_in_vocab
        self.num_classes = n_out_vocab
        self.num_operators = 0

        self.input_vocabulary = {}
        for i in range(n_in_vocab):
            self.input_vocabulary[str(i)] = i

        self.output_vocabulary = {}
        for i in range(n_out_vocab):
            self.output_vocabulary[str(i)] = i

        if operator_vocabulary is not None:
            self.operator_vocabulary = operator_vocabulary
        else:
            self.operator_vocabulary = {}

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
            return ToyDataset.TreeBatch(graph=batched_trees,
                                        mask=batched_trees.ndata['mask'].to(device),
                                        x=batched_trees.ndata['x'].to(device),
                                        y=batched_trees.ndata['y'].to(device))

        return DataLoader(dataset=self, batch_size=batch_size, collate_fn=batcher_dev, shuffle=shuffle,
                          num_workers=0)

    def __get_input_vocab_idx__(self, w):
        # if w not in self.input_vocabulary:
        #     self.input_vocabulary[w] = self.num_vocabs
        #     self.num_vocabs += 1
        return self.input_vocabulary[w]

    def __get_output_vocab_idx__(self, w):
        # if w not in self.output_vocabulary:
        #     self.output_vocabulary[w] = self.num_classes
        #     self.num_classes += 1
        return self.output_vocabulary[w]

    def __get_operator_idx__(self, op):
        if op not in self.operator_vocabulary:
            self.operator_vocabulary[op] = self.num_operators
            self.num_operators += 1
        return self.operator_vocabulary[op]

    def __build_dgl_tree__(self, root):
        g = nx.DiGraph()

        def _rec_build(nid, node):

            n_ch = len(node)-1 #child 0 is always input
            if n_ch > self.max_out_degree:
                self.max_out_degree = n_ch

            if n_ch == 0:
                #leaf
                word = self.__get_input_vocab_idx__(node[0])
                y = self.__get_output_vocab_idx__(node.label())
                g.add_node(nid, x=word, y=y, mask=1, type=-1)
            else:
                # internal
                type = self.__get_operator_idx__(node[0])
                y = self.__get_output_vocab_idx__(node.label())
                g.add_node(nid, x=-1, y=y, mask=0, type=type)
                for i in range(1, n_ch+1):
                    child = node[i]
                    cid = g.number_of_nodes()
                    _rec_build(cid, child)
                    g.add_edge(cid, nid)
        # add root
        _rec_build(0, root)
        ret = dgl.DGLGraph()
        ret.from_networkx(g, node_attrs=['x', 'y', 'mask', 'type'])
        return ret


class TypedAggregator(BaseAggregator):

    def __init__(self, h_size, max_output_degree, pos_stationarity, n_aggr, **kwargs):
        super(TypedAggregator, self).__init__(h_size, max_output_degree, pos_stationarity, n_aggr)

        self.n_type = kwargs['n_type']
        self.cell_list = nn.ModuleList()
        for i in range(self.n_type):
            self.cell_list.append(kwargs['agg_class'](h_size, max_output_degree, pos_stationarity, n_aggr, **kwargs))

    def forward(self, neighbour_h, nodes):

        # get type
        ris = th.zeros((neighbour_h.size(0), self.n_aggr*neighbour_h.size(2)), device=neighbour_h.device)
        for i in range(self.n_type):
            mask = nodes.data['type'] == i
            if th.sum(mask) > 0:
                ris[mask, :] = self.cell_list[i](neighbour_h[mask, :, :], nodes)

        return ris


class ToyOutputModule(nn.Module):

    def __init__(self, h_size, num_classes):
        super(ToyOutputModule, self).__init__()
        self.linear = nn.Linear(h_size, num_classes)

    def forward(self, h):
        return self.linear(h)


def create_toy_model(use_one_hot_encoding, num_vocabs, num_classes, num_operators, out_degree, tree_model_type, x_size, h_size, pos_stationarity,
                     cell_type, rank):

    tree_model_class = get_tree_model_class(tree_model_type)
    agg_class = get_aggregator_class(cell_type)

    if use_one_hot_encoding:
        input_module = nn.Embedding.from_pretrained(th.tril(th.ones((num_vocabs, num_vocabs))), freeze=True)
    else:
        input_module = nn.Embedding(num_vocabs, x_size)

    output_module = ToyOutputModule(h_size, num_classes)

    return tree_model_class(x_size, h_size, out_degree, pos_stationarity, input_module,
                            output_module, TypedAggregator, n_type=num_operators, agg_class=agg_class, rank=rank)


def load_toy_dataset(data_dir):
    data_dir = os.path.join('data', data_dir)
    trainset = ToyDataset(data_dir, ['train.txt'], name='train')
    op_vocab = trainset.operator_vocabulary
    devset = ToyDataset(data_dir, ['dev.txt'], name='dev', operator_vocabulary=op_vocab)
    testset = ToyDataset(data_dir, ['test.txt'], name='test', operator_vocabulary=op_vocab)

    return trainset, devset, testset


def toy_loss_function(output_model, true_label):
    logp = F.log_softmax(output_model, 1)
    return F.nll_loss(logp, true_label, reduction='sum')


def extract_toy_batch_data(batch):
    g = batch.graph
    x = batch.x
    mask = batch.mask
    y = batch.y
    return [g, x, mask], y, g


def toy_single_run_fun(args, device, log_dir):

    if not hasattr(args,'use_one_hot'):
        args.use_one_hot = False

    # load the data
    trainset, devset, testset = load_toy_dataset(args.dataset)

    if args.use_one_hot:
        args.x_size = trainset.num_vocabs
        get_base_logger().info('x_size set to {} due to the one-hot encoding.'.format(trainset.num_vocabs))


    # create the model
    model = create_toy_model(args.use_one_hot, trainset.num_vocabs, trainset.num_classes, trainset.num_operators, trainset.max_out_degree,
                             args.tree_model, args.x_size, args.h_size, args.pos_stationarity,
                             args.cell_type, rank=args.rank).to(device)

    # create the optimizser
    models_params = [x[1] for x in list(model.named_parameters()) if x[1].requires_grad]

    for p in models_params:
        if p.dim() > 1:
            INIT.kaiming_normal_(p)

    optimizer = optim.Adadelta([
        {'params': models_params, 'weight_decay': args.weight_decay}])

    # train and validate
    best_model, info_training = train_and_validate(model, extract_toy_batch_data, toy_loss_function, optimizer, trainset,
                                                   devset, device,
                                                   metrics_class=[Accuracy, RootAccuracy, LeavesAccuracy],
                                                   batch_size=args.batch_size,
                                                   n_epochs=args.epochs, early_stopping_patience=args.early_stopping,
                                                   evaluate_on_training_set=True)

    th.save(best_model, os.path.join(log_dir, 'best_model.pkl'))
    th.save(info_training, os.path.join(log_dir, 'info_training.pkl'))

    # test
    tm, tp = test(best_model, extract_toy_batch_data, testset, device,
                  metrics_class=[Accuracy],
                  batch_size=args.batch_size)

    th.save({'test_metrics': tm, 'test_predictions': tp},
            os.path.join(log_dir, 'test_results.pkl'))

def get_toy_model_selection_fun():
    pass