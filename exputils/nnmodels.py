from collections import OrderedDict
import torch as th
from torch import nn as nn
from exputils.utils import string2class
from exputils.serialisation import from_pkl_file
from exputils.configurations import create_object_from_config


# IMPORTANT!!! DO NOT CHANGE THIS ASSUMPTION
#######################################################
#  in_size and out_size are specified from outside!!  #
#######################################################


class StackableModule(nn.Module):

    def __init__(self, in_size, out_size):
        super(StackableModule, self).__init__()
        if in_size < 0 or out_size < 0:
            raise ValueError("In size and out size cannot be negative!")

        self.in_size = in_size
        self.out_size = out_size

    def forward(self, x):
        pass


class ModuleContainer(StackableModule):

    def forward(self, x):
        return self.container(x)


class Stack(ModuleContainer):

    # TODO: find a way to specify other config. Maybe a dict name_model -> properties of that type of model
    def __init__(self, in_size, out_size, config_modules_list):
        super(Stack, self).__init__(in_size=in_size, out_size=out_size)
        n = len(config_modules_list)
        d = OrderedDict()
        last_out_size = in_size
        for i, module_config in enumerate(config_modules_list):
            if i == n-1:
                # last layer
                m = create_object_from_config(module_config, in_size=last_out_size, out_size=out_size)
            else:
                m = create_object_from_config(module_config, in_size=last_out_size)
            last_out_size = m.out_size
            d[f'layer_{i}'] = m
        self.container = nn.Sequential(d)


class Repeater(ModuleContainer):

    def __init__(self, in_size, out_size, module_config, n_layers, **other_module_config):
        super(Repeater, self).__init__(in_size, out_size)
        d = OrderedDict()
        last_out_size = in_size
        for i in range(n_layers):
            m = create_object_from_config(module_config, in_size=last_out_size, out_size=out_size,
                                          **other_module_config)
            last_out_size = m.out_size
            d[f'layer_{i}'] = m

        self.container = nn.Sequential(d)


class VectorEmbedding(StackableModule):

    def __init__(self, embedding_type, **configs):
        if embedding_type == 'pretrained':
            np_array = from_pkl_file(configs['pretrained_embs'])
            emb = nn.Embedding.from_pretrained(th.tensor(np_array, dtype=th.float), freeze=configs['freeze'])
        elif embedding_type == 'one_hot':
            num_embs = configs['num_embs']
            emb =  nn.Embedding.from_pretrained(th.eye(num_embs, num_embs), freeze=True)
        elif embedding_type == 'random':
            num_embs = configs['num_embs']
            emb_size = configs['emb_size']
            emb = nn.Embedding(num_embs, emb_size)
        else:
            raise ValueError('Embedding type is unkown!')
        super(VectorEmbedding, self).__init__(in_size=1, out_size=emb.embedding_dim)
        self.embedding_layer = emb

    def forward(self, x):
        return self.embedding_layer(x)


class MLP(StackableModule):

    def __init__(self, in_size, out_size, num_hidden_layers=None, h_size=None, h_size_list=None,
                 non_linearity='torch.nn.ReLU', dropout=0):
        super(MLP, self).__init__(in_size, out_size)
        non_linearity_class = string2class(non_linearity)

        if num_hidden_layers is not None and h_size is not None and h_size_list is None:
            # same h_size for each hidden layer
            h_size_list = [h_size] * num_hidden_layers
        elif num_hidden_layers is None and h_size is None and h_size_list is not None:
            # h_size list already specified
            num_hidden_layers = len(h_size_list)
        elif num_hidden_layers == 0 and h_size is None and h_size_list is None:
            # linear layer
            h_size_list = []
        else:
            raise ValueError("H sizes of MLP have not been specified correctly!")

        d = OrderedDict()
        prev_out_size = in_size
        for i in range(num_hidden_layers):
            if dropout > 0:
                d['dropout_{}'.format(i)] = nn.Dropout(dropout)
            d['linear_{}'.format(i)] = nn.Linear(prev_out_size, h_size_list[i])
            d['sigma_{}'.format(i)] = non_linearity_class()
            prev_out_size = h_size_list[i]
        if dropout > 0:
            d['dropout_out'] = nn.Dropout(dropout)
        d['linear_out'] = nn.Linear(prev_out_size, out_size)

        self.MLP = nn.Sequential(d)

    def forward(self, h):
        return self.MLP(h)
