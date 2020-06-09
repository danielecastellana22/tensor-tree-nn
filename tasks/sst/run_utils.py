from experiments.base import Experiment
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import dgl
from preprocessing.utils import ConstValues
from treeRNN.models import TreeModel
from tasks.utils.classifiers import OneLayerNN


class SstExperiment(Experiment):

    def __init__(self, config, output_dir, logger):
        super(SstExperiment, self).__init__(config, output_dir, logger)

    def __get_labels_options__(self):
        dataset_config = self.config.dataset_config
        only_root_labels = dataset_config.only_root_labels if 'only_root_labels' in dataset_config else False
        hide_leaf_labels = dataset_config.hide_leaf_labels if 'hide_leaf_labels' in dataset_config else False

        return only_root_labels, hide_leaf_labels

    def __create_model__(self):
        tree_model_config = self.config.tree_model_config
        output_model_config = self.config.output_model_config
        h_size = tree_model_config.h_size

        input_module = self.__create_input_embedding_module__()
        type_module = self.__create_type_embedding_module__()
        output_module = OneLayerNN(h_size, **output_model_config)

        cell_module = self.__create_cell_module__()

        only_root_labels, hide_leaf_labels = self.__get_labels_options__()

        return TreeModel(input_module, output_module, cell_module, type_module, only_root_state=only_root_labels)

    def __get_loss_function__(self):
        def f(output_model, true_label):
            idxs = (true_label != ConstValues.NO_ELEMENT)
            return F.cross_entropy(output_model[idxs], true_label[idxs], reduction='mean')

        return f

    def __get_batcher_function__(self):
        device = self.__get_device__()
        only_root_labels, hide_leaf_labels = self.__get_labels_options__()

        if hide_leaf_labels:

            def batcher_dev(batch):
                batched_trees = dgl.batch(batch)
                leaves_ids = th.tensor([batched_trees.in_degree(i) == 0 for i in range(batched_trees.number_of_nodes())], dtype=th.bool)
                batched_trees.ndata['y'][leaves_ids] = -1
                batched_trees.to(device)
                return [batched_trees], batched_trees.ndata['y']

        elif only_root_labels:

            def batcher_dev(batch):
                batched_trees = dgl.batch(batch)
                root_ids = th.tensor([batched_trees.out_degree(i) == 0 for i in range(batched_trees.number_of_nodes())], dtype=th.bool)
                y = batched_trees.ndata['y'][root_ids]
                y.to(device)
                batched_trees.to(device)
                return [batched_trees], y
        else:

            def batcher_dev(batch):
                batched_trees = dgl.batch(batch)
                batched_trees.to(device)
                return [batched_trees], batched_trees.ndata['y']

        return batcher_dev