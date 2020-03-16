import os
from utils.experiment import Experiment
from utils.utils import load_embeddings
import torch as th
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as INIT
import torch.nn.functional as F
from treeRNN.dataset import TreeDataset
from treeRNN.metrics import TreeMetric
from treeRNN.models import TreeModel


class SstExperiment(Experiment):

    def __init__(self, id, config, output_dir, logger):
        super(SstExperiment, self).__init__(id, config, output_dir, logger)

    def __load_dataset__(self, load_embs):
        dataset_config = self.config.dataset_config
        dataset_class = dataset_config['dataset_class']
        data_dir = dataset_config['data_dir']

        # load vocabulary
        words_vocab = {'unk': 0}
        with open(os.path.join(data_dir, 'vocab.txt')) as f:
            for l in f.readlines():
                l = l.strip()
                words_vocab[l] = len(words_vocab)

        trainset = dataset_class(data_dir, ['train.pkl'], self.logger.getChild('loading_training_set'), words_vocab=words_vocab)

        args = {'words_vocab': trainset.words_vocab}

        if hasattr(trainset, 'tags_vocab'):
            args['tags_vocab'] = trainset.types_vocab

        if hasattr(trainset, 'types_vocab'):
            args['types_vocab'] = trainset.types_vocab

        valset = dataset_class(data_dir, ['validation.pkl'], self.logger.getChild('loading_validation_set'), **args)
        testset = dataset_class(data_dir, ['test.pkl'], self.logger.getChild('loading_test_set'), **args)

        if load_embs:
            in_pretrained_embs = load_embeddings(data_dir,
                                                 pretrained_embs_file=self.config.input_model_config['pretrained_embs_file'],
                                                 vocab=args['words_vocab'],
                                                 logger=self.logger.getChild('load_embeddings'))
        else:
            in_pretrained_embs = None

        return trainset, valset, testset, in_pretrained_embs, None

    def __create_model__(self, trainset, in_pretrained_embs, type_pretrained_embs):
        tree_model_config = self.config.tree_model_config
        output_model_config = self.config.output_model_config
        x_size = tree_model_config['x_size']
        h_size = tree_model_config['h_size']
        max_out_degree = trainset.max_out_degree
        n_classes = trainset.num_classes
        n_words = trainset.num_words
        n_types = trainset.num_types

        input_module = nn.Embedding.from_pretrained(in_pretrained_embs, freeze=False)
        output_module = SstOutputModule(h_size, n_classes, output_model_config['dropout'])
        type_embs_module = None

        cell_module = self.__create_cell_module__(max_out_degree, n_types)

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
            idxs = (true_label != TreeDataset.NO_ELEMENT)
            return F.cross_entropy(output_model[idxs], true_label[idxs], reduction='sum')

        return f


class SstOutputModule(nn.Module):

    def __init__(self, h_size, num_classes, dropout):
        super(SstOutputModule, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(h_size, num_classes)

    def forward(self, h):
        return self.linear(self.dropout(h))


class RootChildrenAccuracy(TreeMetric):

    def initialise_metric(self):
        self.n_nodes = 0
        self.n_correct = 0

    def update_metric(self, out, gold_label, graph):
        root_ids = [i for i in range(graph.number_of_nodes()) if graph.out_degree(i) == 0]
        root_ch_id = [i for i in range(graph.number_of_nodes()) if i not in root_ids and graph.successors(i).item() in root_ids]

        pred = th.argmax(out, 1)
        self.n_correct += th.sum(th.eq(pred[root_ch_id], gold_label[root_ch_id])).item()
        self.n_nodes += len(root_ch_id)

    def finalise_metric(self):
        self.final_value = self.n_correct / self.n_nodes

    def __str__(self):
        return "Root Children Accuracy: {:4f}".format(self.final_value)

    def is_better_than(self, other_metric):
        return self.final_value > other_metric
