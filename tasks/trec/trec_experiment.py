from experiments.base import Experiment
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import dgl
import numpy as np
from treeRNN.models import TreeModel
from utils.utils import string2class


class TrecExperiment(Experiment):

    def __init__(self, config, output_dir, logger):
        super(TrecExperiment, self).__init__(config, output_dir, logger)

    def __get_num_classes__(self):
        out_type = self.config.dataset_config.output_type
        if out_type == 'coarse':
            return 6
        elif out_type == 'fine':
            return 50
        else:
            raise ValueError('Output type not known!')

    def __create_model__(self):
        tree_model_config = self.config.tree_model_config
        num_classes = self.__get_num_classes__()
        output_model_config = self.config.output_model_config

        h_size = tree_model_config.h_size

        input_module = self.__create_input_embedding_module__()
        type_module = self.__create_type_embedding_module__()
        cell_module = self.__create_cell_module__()

        t = TreeModel(input_module, None, cell_module, type_module)

        output_module = TrecOutputModule(h_size, num_classes, **output_model_config)

        return TrecModel(tree_module=t, output_module=output_module)

    def __get_loss_function__(self):
        def f(output_model, true_label):
            return F.cross_entropy(output_model, true_label, reduction='mean')

        return f

    def __get_batcher_function__(self):
        device = self.__get_device__()
        num_classes = self.__get_num_classes__()

        def batcher_dev(tuple_data):
            tree_list, coarse_label_list, fine_label_list = zip(*tuple_data)
            batched_trees = dgl.batch(tree_list)

            batched_trees.to(device)

            if num_classes == 6:
                out = th.LongTensor(coarse_label_list)
                out.to(device)
            else:
                out = th.LongTensor(fine_label_list)
                out.to(device)

            return [batched_trees], out

        return batcher_dev


class TrecModel(nn.Module):

    def __init__(self, tree_module, output_module):
        super(TrecModel, self).__init__()
        self.tree_module = tree_module
        self.output_module = output_module

    def forward(self, t):
        h_t = self.tree_module(t)

        root_ids = [i for i in range(t.number_of_nodes()) if t.out_degree(i) == 0]

        h_root = h_t[root_ids]

        return self.output_module(h_root)


class TrecOutputModule(nn.Module):

    def __init__(self, in_size, num_classes, dropout, h_size=0, non_linearity='torch.nn.ReLU'):
        super(TrecOutputModule, self).__init__()
        non_linearity_class = string2class(non_linearity)
        if h_size == 0:
            self.MLP = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_size, num_classes))
        else:
            self.MLP = nn.Sequential(nn.Dropout(dropout),
                                     nn.Linear(in_size, h_size), non_linearity_class(), nn.Dropout(dropout),
                                     nn.Linear(h_size, num_classes))

    def forward(self, h):
        return self.MLP(h)