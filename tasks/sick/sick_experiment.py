from experiments.base import Experiment
import torch as th
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import dgl
import numpy as np
from treeRNN.models import TreeModel
from utils.serialization import to_pkl_file
import os
from training.metrics import MSE, Pearson


class SickExperiment(Experiment):

    def __init__(self, config, output_dir, logger):
        super(SickExperiment, self).__init__(config, output_dir, logger)

    def __get_output_type__(self):
        out_type = self.config.dataset_config.output_type
        if out_type == 'relatedness':
            return 0
        elif out_type == 'entailment':
            return 1
        else:
            raise ValueError('Output type not known!')

    def __create_model__(self):
        tree_model_config = self.config.tree_model_config
        output_type = self.__get_output_type__()
        output_model_config = self.config.output_model_config
        x_size = tree_model_config.x_size
        h_size = tree_model_config.h_size

        in_pretrained_embs = self.__load_input_embeddings__()
        input_module = nn.Embedding.from_pretrained(in_pretrained_embs, freeze=True)

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

        t = TreeModel(x_size, h_size, input_module, None, cell_module, type_module)

        if output_type == 0:
            output_module = RelatednessOutputModule(h_size, **output_model_config)
        else:
            output_module = EntailmentOutputModule(h_size, **output_model_config)

        return SickModel(tree_module=t, output_module=output_module)

    def __get_optimiser__(self, model):

        params_to_learn = []
        tree_module = model.tree_module
        output_module = model.output_module
        tree_model_params = list(tree_module.cell_module.parameters()) + list(output_module.parameters())
        params_to_learn.append({'params': tree_model_params,
                                'weight_decay': self.config.tree_model_config['weight_decay']})
                                #'lr': 0.05
                                #'lr_decay': 0.05})

        if tree_module.type_module is not None:
            params_to_learn.append({'params': list(tree_module.type_module.parameters())}) #, 'lr': 0.1})

        #params_to_learn.append({'params': list(tree_module.input_module.parameters())}) #, 'lr': 0.1})

        #create the optimizer
        #optimizer = optim.Adagrad(params_to_learn)
        optimizer = optim.Adadelta(params_to_learn)

        return optimizer

    def __save_test_model_params__(self, best_model):
        type_module = best_model.tree_module.type_module
        if type_module is not None:
            to_pkl_file(type_module.state_dict(), os.path.join(self.output_dir, 'type_embs_learned.pkl'))

    def __get_loss_function__(self):
        output_type = self.__get_output_type__()
        if output_type == 0:
            def f(output_model, true_label):
                return F.kl_div(output_model[0], true_label[0], reduction='batchmean')
        else:
            def f(output_model, true_label):
                return F.cross_entropy(output_model, true_label, reduction='mean')
        return f

    def __get_batcher_function__(self):
        device = self.__get_device__()
        num_classes = self.config.output_model_config.num_classes
        out_type = self.__get_output_type__()

        def batcher_dev(tuple_data):
            a_tree_list, b_tree_list, relatedness_list, entailment_list = zip(*tuple_data)
            batched_a_trees = dgl.batch(a_tree_list)
            batched_b_trees = dgl.batch(b_tree_list)

            batched_a_trees.to(device)
            batched_b_trees.to(device)

            if out_type == 0:
                score_list_th = th.FloatTensor(relatedness_list)
                target_distr = th.FloatTensor(self.get_target_distribution(relatedness_list, num_classes))
                target_distr.to(device)
                score_list_th.to(device)
                out = (target_distr, score_list_th)
            else:
                out = th.LongTensor(entailment_list)
                out.to(device)

            return (batched_a_trees, batched_b_trees), out

        return batcher_dev

    @staticmethod
    def get_target_distribution(labels, num_classes):
        labels = np.array(labels)
        n_el = len(labels)
        target = np.zeros((n_el, num_classes))
        ceil = np.ceil(labels).astype(int)
        floor = np.floor(labels).astype(int)
        idx = (ceil == floor)
        not_idx = np.logical_not(idx)
        target[idx, floor[idx] - 1] = 1
        target[not_idx, floor[not_idx] - 1] = ceil[not_idx] - labels[not_idx]
        target[not_idx, ceil[not_idx] - 1] = labels[not_idx] - floor[not_idx]

        return target


class SickModel(nn.Module):

    def __init__(self, tree_module, output_module):
        super(SickModel, self).__init__()
        self.tree_module = tree_module
        self.output_module = output_module

    def forward(self, g_a, g_b):
        h_a_tree = self.tree_module(g_a)
        h_b_tree = self.tree_module(g_b)

        root_id_a = [i for i in range(g_a.number_of_nodes()) if g_a.out_degree(i) == 0]
        root_id_b = [i for i in range(g_b.number_of_nodes()) if g_b.out_degree(i) == 0]

        h_root_a = h_a_tree[root_id_a]
        h_root_b = h_b_tree[root_id_b]

        return self.output_module(h_root_a, h_root_b)


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


class RelatednessOutputModule(nn.Module):

    def __init__(self, input_dim, hidden_dim, num_classes):
        super(RelatednessOutputModule, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.wh = nn.Linear(2 * self.input_dim, self.hidden_dim)
        self.wp = nn.Linear(self.hidden_dim, self.num_classes)
        self.r = nn.Parameter(th.arange(1, num_classes+1).float().t(), requires_grad=False)

    def forward(self, lvec, rvec):
        mult_dist = th.mul(lvec, rvec)
        abs_dist = th.abs(th.add(lvec, -rvec))
        vec_dist = th.cat((mult_dist, abs_dist), 1)

        distr = th.sigmoid(self.wh(vec_dist))
        distr = F.log_softmax(self.wp(distr), dim=1)

        pred = th.matmul(th.exp(distr), self.r)
        return distr, pred


class MseSICK(MSE):

    def update_metric(self, out, gold_label):
        super(MseSICK, self).update_metric(out[1], gold_label[1])


class PearsonSICK(Pearson):

    def update_metric(self, out, gold_label):
        super(PearsonSICK, self).update_metric(out[1], gold_label[1])
