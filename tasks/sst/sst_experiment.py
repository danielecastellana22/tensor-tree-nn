from experiments.base import Experiment
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import dgl
from preprocessing.utils import ConstValues
from treeRNN.models import TreeModel
from utils.serialization import to_pkl_file
import os


class SstExperiment(Experiment):

    def __init__(self, config, output_dir, logger):
        super(SstExperiment, self).__init__(config, output_dir, logger)

    def __create_model__(self):
        tree_model_config = self.config.tree_model_config
        output_model_config = self.config.output_model_config
        x_size = tree_model_config.x_size
        h_size = tree_model_config.h_size

        in_pretrained_embs = self.__load_input_embeddings__()

        if in_pretrained_embs is not None:
            input_module = nn.Embedding.from_pretrained(in_pretrained_embs, freeze=False)
        else:
            input_module = nn.Embedding(self.config.input_model_config.num_vocabs, x_size)

        output_module = SstOutputModule(h_size, **output_model_config)

        if 'type_model_config' in self.config:
            type_model_config = self.config.type_model_config
            if 'use_pretrained_embs' in type_model_config:
                type_pretrained_embs = self.__load_type_embeddings__()
                type_module = nn.Embedding.from_pretrained(type_pretrained_embs, freeze=False)
            elif 'use_one_hot' in type_model_config and type_model_config.use_one_hot:
                    num_types = type_model_config.num_types
                    type_module = nn.Embedding.from_pretrained(th.eye(num_types, num_types), freeze=True)
            else:
                type_emb_size = self.config.tree_model_config.cell_config.cell_params.type_emb_size
                type_module = nn.Embedding(type_model_config.num_types, type_emb_size)
        else:
            type_module = None

        cell_module = self.__create_cell_module__()

        return TreeModel(x_size, h_size, input_module, output_module, cell_module, type_module)

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
                batch.ndata['y'][leaves_ids] = -1
                batched_trees.to(device)
                return tuple([batched_trees]), batched_trees.ndata['y']
        else:
            def batcher_dev(batch):
                batched_trees = dgl.batch(batch)
                batched_trees.to(device)
                return tuple([batched_trees]), batched_trees.ndata['y']

        return batcher_dev


class SstOutputModule(nn.Module):

    def __init__(self, in_size, num_classes, dropout, h_size=0):
        super(SstOutputModule, self).__init__()
        if h_size == 0:
            self.MLP = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_size, num_classes))
        else:
            self.MLP = nn.Sequential(nn.Dropout(dropout),
                                     nn.Linear(in_size, h_size), nn.ReLU(), nn.Dropout(dropout),
                                     nn.Linear(h_size, num_classes))

    def forward(self, h):
        return self.MLP(h)
