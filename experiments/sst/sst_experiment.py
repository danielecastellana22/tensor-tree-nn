from utils.experiment import Experiment
import torch as th
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as INIT
import torch.nn.functional as F
import dgl
from preprocessing.utils import ConstValues
from treeRNN.models import TreeModel


class SstExperiment(Experiment):

    def __init__(self, config, output_dir, logger):
        super(SstExperiment, self).__init__(config, output_dir, logger)

    def __create_model__(self):
        tree_model_config = self.config.tree_model_config
        output_model_config = self.config.output_model_config
        x_size = tree_model_config.x_size
        h_size = tree_model_config.h_size

        in_pretrained_embs = self.__load_input_embeddings__()

        input_module = nn.Embedding.from_pretrained(in_pretrained_embs, freeze=False)

        output_module = SstOutputModule(h_size, **output_model_config)
        type_embs_module = None

        cell_module = self.__create_cell_module__()

        return TreeModel(x_size, h_size, input_module, output_module, cell_module, type_embs_module)

    def __get_optimiser__(self, model):
        params_trees = [x[1] for x in list(model.named_parameters()) if
                        'input_module' not in x[0]]
        params_emb = list(model.input_module.parameters())

        # TODO: use set initializer
        for p in params_trees:
            if p.dim() > 1:
                INIT.xavier_uniform_(p)

        # create the optimizer
        # optimizer = optim.Adagrad([
        #     {'params': params_trees, 'lr': 0.05, 'weight_decay': args.weight_decay},#, 'lr_decay': 0.001},
        #     {'params': params_emb, 'lr': 0.1}])

        optimizer = optim.Adadelta([
            {'params': params_trees, 'weight_decay': self.config.tree_model_config['weight_decay']},
            {'params': params_emb}])

        return optimizer

    def __get_loss_function__(self):
        def f(output_model, true_label):
            idxs = (true_label != ConstValues.NO_ELEMENT)
            return F.cross_entropy(output_model[idxs], true_label[idxs], reduction='sum')

        return f

    def __get_batcher_function__(self):
        device = self.__get_device__()

        def batcher_dev(batch):
            batched_trees = dgl.batch(batch)
            batched_trees.to(device)
            return tuple([batched_trees]), batched_trees.ndata['y']

        return batcher_dev


class SstOutputModule(nn.Module):

    def __init__(self, h_size, num_classes, dropout):
        super(SstOutputModule, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(h_size, num_classes)

    def forward(self, h):
        return self.linear(self.dropout(h))
