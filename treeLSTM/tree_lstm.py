import torch as th
import torch.nn as nn
import dgl


# TODO: maybe is reasonable a flag for positional stationarity
# The Full aggregator is available only when maxOutput Degree is 2
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


# h = U1*h1 + U2*h2 + ... + Un*hn
class NaryAggregator(nn.Module):
    def __init__(self, in_size, out_size, max_output_degree, **kwargs):
        super(NaryAggregator, self).__init__()

        self.U = nn.Linear(max_output_degree * in_size, out_size, kwargs)

    # neighbour_states has shape batch_size x n_neighbours x insize
    def forward(self, neighbour_states):
        h_cat = neighbour_states.view(neighbour_states.size(0), -1)
        return self.U(h_cat)


# h = h1 + h2 + ... + hn
class SumChildAggregator(nn.Module):
    def __init__(self):
        super(SumChildAggregator, self).__init__()

    # neighbour_states has shape batch_size x n_neighbours x insize
    def forward(self, neighbour_states):
        return th.sum(neighbour_states, 1)


# todo: what about bias
# h = U3*r3, where r3 = G*r1*r2, r1 = U1*h1 and r2 = U2*h2
class HOSVDAggregator(nn.Module):
    def __init__(self, in_size, out_size, max_output_degree, rank):
        super(HOSVDAggregator, self).__init__()

        self.max_output_degree = max_output_degree
        # core tensor
        sz_G = tuple(rank for i in range(max_output_degree+1))
        self.G = nn.Parameter(th.rand(sz_G))

        # mode matrices
        self.U_list = []
        for i in range(max_output_degree):
            self.U_list.append(nn.Linear(in_size, rank, bias=True))

        self.U_output = nn.Linear(rank, out_size, bias=False)

    # neighbour_states has shape batch_size x n_neighbours x insize
    def forward(self, neighbour_states):
        ein_str1 = ''
        ein_str2 = ''
        offset = ord('a')
        operand_list = [self.G]
        for i in range(self.max_output_degree):
            h = neighbour_states[:, i, :].view(neighbour_states.size(0), -1)
            U = self.U_list[i]
            operand_list.append(U(h))
            cc = chr(offset+i)
            ein_str1 += cc
            ein_str2 += 'z'+cc
            if i < self.max_output_degree-1:
                ein_str2 += ','

        out_cc = chr(offset+self.max_output_degree)
        r_out = th.einsum(ein_str1 + ',' + ein_str2 + '->n'+out_cc, operand_list)
        h_out = self.U_output(r_out)

        return h_out


# h3 =  Canonical decomposition
class CANCOMPAggregator(nn.Module):
    def __init__(self, in_size, out_size, max_output_degree, rank):
        super(CANCOMPAggregator, self).__init__()

        # CANCOMP matrices
        self.max_output_degree = max_output_degree

        # mode matrices
        self.U_list = []
        for i in range(max_output_degree):
            self.U_list.append(nn.Linear(in_size, rank, bias=True))

        self.U_output = nn.Linear(rank, out_size, bias=False)

    # neighbour_states has shape batch_size x n_neighbours x insize
    def forward(self, neighbour_states):
        ein_str = ''
        operand_list = []
        for i in range(self.max_output_degree):
            h = neighbour_states[:, i, :].view(neighbour_states.size(0), -1)
            U = self.U_list[i]
            operand_list.append(U(h))

            ein_str += 'nr,'

        ein_str += 'rk->nk'

        operand_list.append(self.U_output)
        h_out = th.einsum(ein_str, operand_list)

        return h_out


class TTAggregator(nn.Module):
    def __init__(self, in_size, out_size, max_output_degree, rank):
        super(TTAggregator, self).__init__()

        self.max_output_degree = max_output_degree

        # TT tensors
        self.U_list = []
        self.U_list.append(nn.Parameter(th.rand((in_size, rank))))
        for i in range(max_output_degree):
            self.U_list.append(nn.Parameter(th.rand((in_size, rank, rank))))

        self.U_output = nn.Parameter(th.rand((rank, out_size)))

    # neighbour_states has shape batch_size x n_neighbours x insize
    def forward(self, neighbour_states):

        h = neighbour_states[:, 0, :].view(neighbour_states.size(0), -1)
        U = self.U_list[0]
        ris = th.einsum('bc,nb->nc',U,h)
        for i in range(1,self.max_output_degree):
            h = neighbour_states[:, i, :].view(neighbour_states.size(0), -1)
            U = self.U_list[i]
            ris = th.einsum('abc,na,nb->nc',U,ris,h)

        h_out = th.einsum('ab,na->nb',self.U_output, ris)

        return h_out


class GenericTreeLSTMCell(nn.Module):
    def __init__(self, x_size, h_size, max_output_degree, cell_type, **cell_args):
        super(GenericTreeLSTMCell, self).__init__()
        # for the input
        self.W_iou = nn.Linear(x_size, 3 * h_size, bias=False)
        self.b_iou = nn.Parameter(th.zeros(1, 3 * h_size))

        if cell_type == 'nary':
            self.f_aggregator = NaryAggregator(h_size, 2 * h_size, max_output_degree, bias=False)
            self.iou_aggregator = NaryAggregator(h_size, 3 * h_size, max_output_degree, bias=True)
        elif cell_type == 'sum':
            self.f_aggregator = SumChildAggregator()
            self.iou_aggregator = self.f_aggregator
        elif cell_type == 'hosvd':
            self.f_aggregator = HOSVDAggregator(h_size, 2*h_size, max_output_degree, **cell_args)
            self.iou_aggregator = HOSVDAggregator(h_size, 3*h_size, max_output_degree, **cell_args)
        elif cell_type == 'tt':
            self.f_aggregator = TTAggregator(h_size, 2*h_size, max_output_degree, **cell_args)
            self.iou_aggregator = TTAggregator(h_size, 3*h_size, max_output_degree, **cell_args)
        elif cell_type == 'cd':
            self.f_aggregator = CANCOMPAggregator(h_size, 2 * h_size, max_output_degree, **cell_args)
            self.iou_aggregator = CANCOMPAggregator(h_size, 3 * h_size, max_output_degree, **cell_args)
        elif cell_type == 'full':
            if max_output_degree > 2:
                raise ValueError('Full cel type can be use only with a maximum output degree of 2')
            self.f_aggregator = BinaryFullTensorAggregator(h_size, 2*h_size)
            self.iou_aggregator = BinaryFullTensorAggregator(h_size, 3*h_size)

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


class TreeLSTM(nn.Module):
    def __init__(self,
                 x_size,
                 h_size,
                 max_output_degree,
                 input_module,
                 output_module,
                 cell_type='nary', **cell_args):
        super(TreeLSTM, self).__init__()
        self.x_size = x_size
        self.h_size = h_size
        self.input_module = input_module
        self.output_module = output_module
        self.cell = GenericTreeLSTMCell(x_size, h_size, max_output_degree, cell_type, **cell_args)

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
        embeds = self.input_module(batch.x * batch.mask)
        g.ndata['iou'] = self.cell.W_iou(embeds) * batch.mask.float().unsqueeze(-1)
        g.ndata['h'] = h
        g.ndata['c'] = c
        # propagate
        dgl.prop_nodes_topo(g)
        # compute output
        h = g.ndata.pop('h')
        out = self.output_module(h)
        return out
