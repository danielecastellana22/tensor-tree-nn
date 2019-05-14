import torch as th
import torch.nn as nn
import dgl


#TODO: this should be a particular case of tree_lstm
# h = U1*h1 + U2*h2
class BinaryNaryAggregator(nn.Module):
    def __init__(self, in_size, out_size, **kwargs):
        super(BinaryNaryAggregator, self).__init__()

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


class BinaryGenericTreeLSTMCell(nn.Module):
    def __init__(self, x_size, h_size, cell_type, **cell_args):
        super(BinaryGenericTreeLSTMCell, self).__init__()
        self.W_iou = nn.Linear(x_size, 3 * h_size, bias=False)
        self.b_iou = nn.Parameter(th.zeros(1, 3 * h_size))
        if cell_type == 'nary':
            self.f_aggregator = BinaryNaryAggregator(h_size, 2 * h_size, bias=False)
            self.iou_aggregator = BinaryNaryAggregator(h_size, 3 * h_size, bias=True)
        elif cell_type == 'sum':
            self.f_aggregator = SumChildAggregator()
            self.iou_aggregator = self.f_aggregator
        elif cell_type == 'full':
            self.f_aggregator = BinaryFullTensorAggregator(h_size, 2*h_size)
            self.iou_aggregator = BinaryFullTensorAggregator(h_size, 3*h_size)
        elif cell_type == 'hosvd':
            self.f_aggregator = BinaryHOSVDAggregator(h_size, 2*h_size, cell_args)
            self.iou_aggregator = BinaryHOSVDAggregator(h_size, 3*h_size, cell_args)

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


class BinaryTreeLSTM(nn.Module):
    def __init__(self,
                 x_size,
                 h_size,
                 input_module,
                 output_module,
                 cell_type='nary', **cell_args):
        super(BinaryTreeLSTM, self).__init__()
        self.x_size = x_size
        self.h_size = h_size
        self.input_module = input_module
        self.output_module = output_module
        self.cell = BinaryGenericTreeLSTMCell(x_size, h_size, cell_type, **cell_args)

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
        embeds = self.embedding(batch.x * batch.mask)
        g.ndata['iou'] = self.cell.W_iou(embeds) * batch.mask.float().unsqueeze(-1)
        g.ndata['h'] = h
        g.ndata['c'] = c
        # propagate
        dgl.prop_nodes_topo(g)
        # compute output
        h = g.ndata.pop('h')
        out = self.output_module(h)
        return out
