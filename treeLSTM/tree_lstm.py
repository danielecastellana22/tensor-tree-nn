"""
Improved Semantic Representations From Tree-Structured Long Short-Term Memory Networks
https://arxiv.org/abs/1503.00075
"""
import time
import itertools
import networkx as nx
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import dgl


# h = U1*h1 + U2*h2
class NaryAggregator(nn.Module):
    def __init__(self, in_size, out_size, **kwargs):
        super(NaryAggregator, self).__init__()

        self.U = nn.Linear(2 * in_size, out_size, kwargs)

    # neighbour_states has shape batch_size x n_neighbours x insize
    def forward(self, neighbour_states):
        h_cat = neighbour_states.view(neighbour_states.size(0), -1)
        return self.U(h_cat)


# h = h1 + h2
class SumChildAggregator(nn.Module):
    def __init__(self):
        super(SumChildAggregator, self).__init__()

    # neighbour_states has shape batch_size x n_neighbours x insize
    def forward(self, neighbour_states):
        return th.sum(neighbour_states, 1)


# h = A*h1*h2 + U1*h1 + U2*h2 + b
class BinaryFullTensorAggregator(nn.Module):
    def __init__(self, in_size, out_size):
        super(BinaryFullTensorAggregator, self).__init__()

        self.A = nn.Parameter(th.rand(in_size, in_size, out_size))
        self.U1 = nn.Linear(in_size, out_size, bias=False)
        self.U2 = nn.Linear(in_size, out_size, bias=False)
        self.b = nn.Parameter(th.rand(out_size))

    # neighbour_states has shape batch_size x n_neighbours x insize
    def forward(self, neighbour_states):
        h1 = neighbour_states[:, 0, :].view(neighbour_states.size(0), -1)
        h2 = neighbour_states[:, 1, :].view(neighbour_states.size(0), -1)
        return th.einsum('ijk,ni,nj->nk', self.A, h1, h2) + self.U1(h1) + self.U2(h2) + self.b


# h = U3*r3, where r3 = G*r1*r2, r1 = U1*h1 and r2 = U2*h2
class BinaryHOSVDAggregator(nn.Module):
    def __init__(self, in_size, out_size, rank):
        super(BinaryHOSVDAggregator, self).__init__()

        # core tensor
        self.G = nn.Parameter(th.rand(rank, rank, rank))

        # mode matrices
        self.U1 = nn.Linear(in_size, rank, bias=True)
        self.U2 = nn.Linear(in_size, rank, bias=True)
        self.U3 = nn.Linear(rank, out_size, bias=False)

    # neighbour_states has shape batch_size x n_neighbours x insize
    def forward(self, neighbour_states):
        h1 = neighbour_states[:, 0, :].view(neighbour_states.size(0), -1)
        h2 = neighbour_states[:, 1, :].view(neighbour_states.size(0), -1)

        r1 = self.U1(h1)
        r2 = self.U2(h2)
        r3 = th.einsum('ijk,ni,nj->nk', self.G, r1, r2)
        h3 = self.U3(r3)

        return h3


# h3 =  Canonical decomposition
class BinaryCANCOMPAggregator(nn.Module):
    def __init__(self, in_size, out_size, rank):
        super(BinaryCANCOMPAggregator, self).__init__()

        # CANCOMP matrices
        self.U1 = nn.Linear(in_size, rank, bias=True)
        self.U2 = nn.Linear(in_size, rank, bias=True)
        self.U3 = nn.Linear(rank, out_size, bias=False)

    # neighbour_states has shape batch_size x n_neighbours x insize
    def forward(self, neighbour_states):
        h1 = neighbour_states[:, 0, :].view(neighbour_states.size(0), -1)
        h2 = neighbour_states[:, 1, :].view(neighbour_states.size(0), -1)

        r1 = self.U1(h1)
        r2 = self.U2(h2)
        h3 = th.einsum('rk,nr,nr->nk', self.U3, r1, r2)

        return h3


class GenericTreeLSTMCell(nn.Module):
    def __init__(self, x_size, h_size, cell_type, **cell_args):
        super(GenericTreeLSTMCell, self).__init__()
        self.W_iou = nn.Linear(x_size, 3 * h_size, bias=False)
        self.b_iou = nn.Parameter(th.zeros(1, 3 * h_size))
        if cell_type == 'nary':
            self.f_aggregator = NaryAggregator(h_size, 2 * h_size, bias=False)
            self.iou_aggregator = NaryAggregator(h_size, 3 * h_size, bias=True)
        elif cell_type == 'sum':
            self.f_aggregator = SumChildAggregator()
            self.iou_aggregator = self.f_aggregator
        elif cell_type == 'full':
            self.f_aggregator = BinaryFullTensorAggregator(h_size, 2*h_size)
            self.iou_aggregator = BinaryFullTensorAggregator(h_size, 3*h_size)
        elif cell_type == 'hosvd':
            # TODO: set rank from main
            self.f_aggregator = BinaryHOSVDAggregator(h_size, 2*h_size, rank=20)
            self.iou_aggregator = BinaryHOSVDAggregator(h_size, 3*h_size, rank=20)

    def message_func(self, edges):
        return {'h': edges.src['h'], 'c': edges.src['c']}

    def reduce_func(self, nodes):
        f_aggr = self.f_aggregator(nodes.mailbox['h'])
        iou_aggr = self.iou_aggregator(nodes.mailbox['h'])

        f = th.sigmoid(f_aggr).view(*nodes.mailbox['h'].size())
        c = th.sum(f * nodes.mailbox['c'], 1)
        return {'iou': iou_aggr, 'c': c}

    def apply_node_func(self, nodes):
        iou = nodes.data['iou'] + self.b_iou
        i, o, u = th.chunk(iou, 3, 1)
        i, o, u = th.sigmoid(i), th.sigmoid(o), th.tanh(u)
        c = i * u + nodes.data['c']
        h = o * th.tanh(c)
        return {'h' : h, 'c' : c}


class TreeLSTMCell(nn.Module):
    def __init__(self, x_size, h_size):
        super(TreeLSTMCell, self).__init__()
        self.W_iou = nn.Linear(x_size, 3 * h_size, bias=False)
        self.U_iou = nn.Linear(2 * h_size, 3 * h_size, bias=False)
        self.b_iou = nn.Parameter(th.zeros(1, 3 * h_size))
        self.U_f = nn.Linear(2 * h_size, 2 * h_size)

    def message_func(self, edges):
        return {'h': edges.src['h'], 'c': edges.src['c']}

    def reduce_func(self, nodes):
        h_cat = nodes.mailbox['h'].view(nodes.mailbox['h'].size(0), -1)
        f = th.sigmoid(self.U_f(h_cat)).view(*nodes.mailbox['h'].size())
        c = th.sum(f * nodes.mailbox['c'], 1)
        return {'iou': self.U_iou(h_cat), 'c': c}

    def apply_node_func(self, nodes):
        iou = nodes.data['iou'] + self.b_iou
        i, o, u = th.chunk(iou, 3, 1)
        i, o, u = th.sigmoid(i), th.sigmoid(o), th.tanh(u)
        c = i * u + nodes.data['c']
        h = o * th.tanh(c)
        return {'h' : h, 'c' : c}


class ChildSumTreeLSTMCell(nn.Module):
    def __init__(self, x_size, h_size):
        super(ChildSumTreeLSTMCell, self).__init__()
        self.W_iou = nn.Linear(x_size, 3 * h_size, bias=False)
        self.U_iou = nn.Linear(h_size, 3 * h_size, bias=False)
        self.b_iou = nn.Parameter(th.zeros(1, 3 * h_size))
        self.U_f = nn.Linear(h_size, h_size)

    def message_func(self, edges):
        return {'h': edges.src['h'], 'c': edges.src['c']}

    def reduce_func(self, nodes):
        h_tild = th.sum(nodes.mailbox['h'], 1)
        f = th.sigmoid(self.U_f(nodes.mailbox['h']))
        c = th.sum(f * nodes.mailbox['c'], 1)
        return {'iou': self.U_iou(h_tild), 'c': c}

    def apply_node_func(self, nodes):
        iou = nodes.data['iou'] + self.b_iou
        i, o, u = th.chunk(iou, 3, 1)
        i, o, u = th.sigmoid(i), th.sigmoid(o), th.tanh(u)
        c = i * u + nodes.data['c']
        h = o * th.tanh(c)
        return {'h': h, 'c': c}


# TODO: add general superclass TreeLSTM with input/output module modifiable
class TreeLSTM(nn.Module):
    def __init__(self,
                 num_vocabs,
                 x_size,
                 h_size,
                 num_classes,
                 dropout,
                 pretrained_emb=None,
                 cell_type='nary', **cell_args):
        super(TreeLSTM, self).__init__()
        self.x_size = x_size
        self.h_size = h_size
        self.num_classes = num_classes
        self.embedding = nn.Embedding(num_vocabs, x_size)
        if pretrained_emb is not None:
            self.embedding.weight.data.copy_(pretrained_emb)
            self.embedding.weight.requires_grad = True
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(h_size, num_classes)
        self.cell = GenericTreeLSTMCell(x_size, h_size, cell_type, **cell_args)
        #cell = GenericTreeLSTMCell
        #self.cell = cell(x_size, h_size)

    def forward(self, batch, h, c):
        """Compute tree-lstm prediction given a batch.
        Parameters
        ----------
        batch : dgl.data.SSTBatch
            The data batch.
        h : Tensor
            Initial hidden state.
        c : Tensor
            Initial cell state.
        Returns
        -------
        logits : Tensor
            The prediction of each node.
        """
        g = batch.graph
        g.register_message_func(self.cell.message_func)
        g.register_reduce_func(self.cell.reduce_func)
        g.register_apply_node_func(self.cell.apply_node_func)
        # feed embedding
        embeds = self.embedding(batch.wordid * batch.mask)
        g.ndata['iou'] = self.cell.W_iou(embeds) * batch.mask.float().unsqueeze(-1)
        g.ndata['h'] = h
        g.ndata['c'] = c
        # propagate
        dgl.prop_nodes_topo(g)
        # compute logits
        #h = self.dropout(g.ndata.pop('h'))
        h = g.ndata.pop('h')
        logits = self.linear(h)
        return logits
