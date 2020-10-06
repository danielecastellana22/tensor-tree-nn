import os
from tqdm import tqdm
from experiments.base import CollateFun
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import dgl
import numpy as np
from experiments.metrics import MSE, Pearson
from preprocessing.base import NlpParsedTreesPreprocessor
from utils.serialization import from_pkl_file, to_pkl_file


class SickParsedTreesPreprocessor(NlpParsedTreesPreprocessor):

    def __init__(self, config):
        super(SickParsedTreesPreprocessor, self).__init__(config)

    def preprocess(self):
        config = self.config
        input_dir = config.input_dir
        output_dir = config.output_dir

        tree_type = config.preprocessor_config.tree_type

        # set file names
        file_names = {'train': ['SICK_train_{}.pkl'.format(x) for x in tree_type],
                      'validation': ['SICK_trial_{}.pkl'.format(x) for x in tree_type],
                      'test': ['SICK_test_{}.pkl'.format(x) for x in tree_type]}

        # preprocessing trees
        for tag_name, f_list in file_names.items():
            parsed_trees_list = []
            for f in f_list:
                parsed_trees_list.append(from_pkl_file(os.path.join(input_dir, f)))

            n_trees = len(parsed_trees_list[0])
            parsed_trees = [{'tree_a': tuple([v[i]['tree_a'] for v in parsed_trees_list]),
                             'tree_b': tuple([v[i]['tree_b'] for v in parsed_trees_list]),
                             'relatedness': parsed_trees_list[0][i]['relatedness'],
                             'entailment': parsed_trees_list[0][i]['entailment']} for i in range(n_trees)]

            self.__init_stats__(tag_name)

            data_list = []

            for x in tqdm(parsed_trees, desc='Preprocessing {}'.format(tag_name)):
                t_a = self.tree_transformer.transform(*x['tree_a'])
                t_b = self.tree_transformer.transform(*x['tree_b'])

                self.__assign_node_features__(t_a)
                self.__assign_node_features__(t_b)

                self.__update_stats__(tag_name, t_a)
                self.__update_stats__(tag_name, t_b)

                dgl_t_a = self.__nx_to_dgl__(t_a)
                dgl_t_b = self.__nx_to_dgl__(t_b)
                data_list.append((dgl_t_a, dgl_t_b, x['relatedness'], x['entailment']))

            self.__print_stats__(tag_name)
            to_pkl_file(data_list, os.path.join(output_dir, '{}.pkl'.format(tag_name)))

        # save all stats
        self.__save_stats__()

        self.__save_word_embeddings__()


class SickKlLoss:

    def __call__(self, output_model, true_label):
        return F.kl_div(output_model[0], true_label[0], reduction='batchmean')


class SickCollateFun(CollateFun):

    def __init__(self, output_type, device):
        super(SickCollateFun, self).__init__(device)
        if output_type == 'relatedness':
            self.output_type = 0
        elif output_type == 'entailment':
            self.output_type = 1
        else:
            raise ValueError('Output type not known!')

    def __call__(self, tuple_data):
        a_tree_list, b_tree_list, relatedness_list, entailment_list = zip(*tuple_data)
        batched_a_trees = dgl.batch(a_tree_list)
        batched_b_trees = dgl.batch(b_tree_list)

        batched_a_trees.to(self.device)
        batched_b_trees.to(self.device)

        if self.output_type == 0:
            score_list_th = th.tensor(relatedness_list)
            target_distr = th.tensor(self.get_target_distribution(relatedness_list, 5))
            target_distr.to(self.device)
            score_list_th.to(self.device)
            out = (target_distr, score_list_th)
        else:
            out = th.tensor(entailment_list, dtype=th.long)
            out.to(self.device)

        return (batched_a_trees, batched_b_trees), out

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


class EntailmentClassifier(nn.Module):

    def __init__(self, input_dim, hidden_dim, num_classes):
        super(EntailmentClassifier, self).__init__()
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


class RelatednessClassifier(nn.Module):

    def __init__(self, input_dim, hidden_dim, num_classes):
        super(RelatednessClassifier, self).__init__()
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