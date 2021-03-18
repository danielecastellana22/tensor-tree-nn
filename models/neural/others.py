from collections import OrderedDict
import torch as th
from torch import nn as nn
from exputils.utils import string2class
from exputils.serialisation import from_pkl_file


class VectorEmbedding(nn.Module):

    def __new__(cls, *args, **kwargs):
        embedding_type = kwargs['embedding_type']
        if embedding_type == 'pretrained':
            np_array = from_pkl_file(kwargs['pretrained_embs'])
            return nn.Embedding.from_pretrained(th.tensor(np_array, dtype=th.float), freeze=kwargs['freeze'])
        elif embedding_type == 'one_hot':
            num_embs = kwargs['num_embs']
            return nn.Embedding.from_pretrained(th.eye(num_embs, num_embs), freeze=True)
        elif embedding_type == 'random':
            num_embs = kwargs['num_embs']
            emb_size = kwargs['emb_size']
            return nn.Embedding(num_embs, emb_size)
        else:
            raise ValueError('Embedding type is unkown!')


class MLP(nn.Module):

    def __init__(self, in_size, out_size, dropout=0, num_layers=0, h_size=-1, non_linearity='torch.nn.ReLU'):
        super(MLP, self).__init__()
        non_linearity_class = string2class(non_linearity)

        d = OrderedDict()
        prev_out_size = in_size
        for i in range(num_layers):
            if dropout > 0:
                d['dropout_{}'.format(i)] = nn.Dropout(dropout)
            d['linear_{}'.format(i)] = nn.Linear(prev_out_size, h_size)
            d['sigma_{}'.format(i)] = non_linearity_class()
            prev_out_size = h_size
        if dropout > 0:
            d['dropout_out'] = nn.Dropout(dropout)
        d['linear_out'] = nn.Linear(prev_out_size, out_size)

        self.MLP = nn.Sequential(d)

    def forward(self, h):
        return self.MLP(h)