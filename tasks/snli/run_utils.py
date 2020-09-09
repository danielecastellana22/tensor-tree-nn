from experiments.base import Experiment
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import dgl
from treeRNN.models import TreeModel
from training.metrics import MSE, Pearson
from utils.serialization import from_pkl_file
from torch.utils.data import ConcatDataset
from preprocessing.dataset import ListDataset
import os

class SnliExperiment(Experiment):

    def __init__(self, config, output_dir, logger):
        super(SnliExperiment, self).__init__(config, output_dir, logger)

    def __load_training_data__(self):
        dataset_config = self.config.dataset_config

        data_dir = dataset_config.data_dir

        tr_list = []
        for i in range(12):#2):
            self.logger.debug('READING TRAINING SET {}'.format(i))
            tr_list.append(ListDataset(from_pkl_file(os.path.join(data_dir, 'train_{}.pkl'.format(i)))))
        trainset = ConcatDataset(tr_list)
        valset = ListDataset(from_pkl_file(os.path.join(data_dir, 'validation_0.pkl')))

        return trainset, valset

    def __load_test_data__(self):
        dataset_config = self.config.dataset_config
        data_dir = dataset_config.data_dir

        testset = ListDataset(from_pkl_file(os.path.join(data_dir, 'test_0.pkl')))

        return testset

    def __create_model__(self):
        tree_model_config = self.config.tree_model_config
        output_model_config = self.config.output_model_config

        h_size = tree_model_config.h_size

        input_module = self.__create_input_embedding_module__()
        type_module = self.__create_type_embedding_module__()
        cell_module = self.__create_cell_module__()

        output_module = EntailmentOutputModule(h_size, **output_model_config)

        return TreeModel(input_module, output_module, cell_module, type_module, only_root_state=True)

    def __get_loss_function__(self):
        def f(output_model, true_label):
            return F.cross_entropy(output_model, true_label, reduction='mean')

        return f

    def __get_batcher_function__(self):
        device = self.__get_device__()
        # num_classes = self.config.output_model_config.num_classes

        def batcher_dev(tuple_data):
            a_tree_list, b_tree_list, entailment_list = zip(*tuple_data)
            batched_a_trees = dgl.batch(a_tree_list)
            batched_b_trees = dgl.batch(b_tree_list)

            batched_a_trees.to(device)
            batched_b_trees.to(device)

            out = th.LongTensor(entailment_list)
            out.to(device)

            return (batched_a_trees, batched_b_trees), out

        return batcher_dev


class EntailmentOutputModule(nn.Module):

    def __init__(self, input_dim, hidden_dim, num_classes):
        super(EntailmentOutputModule, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.wh = nn.Linear(2 * self.input_dim, self.hidden_dim)
        self.wp = nn.Linear(self.hidden_dim, self.num_classes)

    def forward(self, lvec, rvec):
        mult_dist = th.mul(lvec, rvec)
        abs_dist = th.abs(th.add(lvec, -rvec))
        vec_dist = th.cat((mult_dist, abs_dist), 1)

        distr = th.tanh(self.wh(vec_dist))
        distr = self.wp(distr)

        return distr


class MseSICK(MSE):

    def update_metric(self, out, gold_label):
        super(MseSICK, self).update_metric(out[1], gold_label[1])


class PearsonSICK(Pearson):

    def update_metric(self, out, gold_label):
        super(PearsonSICK, self).update_metric(out[1], gold_label[1])
