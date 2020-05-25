from experiments.base import Experiment
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import dgl
from preprocessing.utils import ConstValues
from treeRNN.models import TreeModel
from utils.serialization import to_pkl_file
from utils.utils import string2class
import os


class SstExperiment(Experiment):

    def __init__(self, config, output_dir, logger):
        super(SstExperiment, self).__init__(config, output_dir, logger)

    def __create_model__(self):
        tree_model_config = self.config.tree_model_config
        output_model_config = self.config.output_model_config
        h_size = tree_model_config.h_size

        input_module = self.__create_input_embedding_module__()
        type_module = self.__create_type_embedding_module__()
        output_module = SstOutputModule(h_size, **output_model_config)

        cell_module = self.__create_cell_module__()

        return TreeModel(input_module, output_module, cell_module, type_module)

    def __save_test_model_params__(self, best_model):
        to_pkl_file(best_model.state_dict(), os.path.join(self.output_dir, 'params_learned.pkl'))

    def __get_loss_function__(self):

        def f(output_model, true_label):
            idxs = (true_label != ConstValues.NO_ELEMENT)
            return F.cross_entropy(output_model[idxs], true_label[idxs], reduction='mean')

        return f

    def __get_batcher_function__(self):
        device = self.__get_device__()
        hide_leaf_labels = self.config.dataset_config.hide_leaf_labels

        if hide_leaf_labels:
            def batcher_dev(batch):
                batched_trees = dgl.batch(batch)
                leaves_ids = th.BoolTensor([batched_trees.in_degree(i) == 0 for i in range(batched_trees.number_of_nodes())])
                batched_trees.ndata['y'][leaves_ids] = -1
                batched_trees.to(device)
                return tuple([batched_trees]), batched_trees.ndata['y']
        else:
            def batcher_dev(batch):
                batched_trees = dgl.batch(batch)
                batched_trees.to(device)
                return tuple([batched_trees]), batched_trees.ndata['y']

        return batcher_dev


class SstOutputModule(nn.Module):

    def __init__(self, in_size, num_classes, dropout, h_size=0, non_linearity='torch.nn.ReLU'):
        super(SstOutputModule, self).__init__()
        non_linearity_class = string2class(non_linearity)
        if h_size == 0:
            self.MLP = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_size, num_classes))
        else:
            self.MLP = nn.Sequential(nn.Dropout(dropout),
                                     nn.Linear(in_size, h_size), non_linearity_class(), nn.Dropout(dropout),
                                     nn.Linear(h_size, num_classes))

    def forward(self, h):
        return self.MLP(h)
