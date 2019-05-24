import torch as th
import torch.nn as nn
import dgl
import torch.nn.functional as F

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
        self.max_output_degree = max_output_degree
        self.U = nn.Linear(max_output_degree * in_size, out_size, kwargs)

    # neighbour_states has shape batch_size x n_neighbours x insize
    def forward(self, neighbour_states):
        # no missing children
        h_cat = neighbour_states.view(neighbour_states.size(0), -1)
        return self.U(h_cat)

# h = h1 + h2 + ... + hn
class SumChildAggregator(nn.Module):
    def __init__(self):
        super(SumChildAggregator, self).__init__()

    # neighbour_states has shape batch_size x n_neighbours x insize
    def forward(self, neighbour_states):
        return th.sum(neighbour_states, 1)


# TODO: what about bias
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
        for i in range(1, self.max_output_degree):
            h = neighbour_states[:, i, :].view(neighbour_states.size(0), -1)
            U = self.U_list[i]
            ris = th.einsum('abc,na,nb->nc',U,ris,h)

        h_out = th.einsum('ab,na->nb',self.U_output, ris)

        return h_out


class GenericTreeLSTMCell(nn.Module):

    def __init__(self, x_size, h_size, max_output_degree, cell_type, **cell_args):
        super(GenericTreeLSTMCell, self).__init__()
        # TODO: add parameter to choose to freeze or not the bottom values
        self.bottom_h = nn.Parameter(th.zeros(h_size), requires_grad=False)
        self.bottom_c = nn.Parameter(th.zeros(h_size), requires_grad=False)

        # for the input
        self.max_output_degree = max_output_degree

        self.W_iou = nn.Linear(x_size, 3 * h_size, bias=False)
        self.b_iou = nn.Parameter(th.zeros(1, 3 * h_size))
        # TODO: 2 must be replaced by max_output_degree
        self.W_f = nn.Linear(x_size, h_size, bias=False)
        self.b_f = nn.Parameter(th.zeros(1, h_size))

        if cell_type == 'nary':
            self.f_aggregator = NaryAggregator(h_size, 2 * h_size, max_output_degree, bias=False)
            self.iou_aggregator = NaryAggregator(h_size, 3 * h_size, max_output_degree, bias=False)
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

    def forward(self, *input):
        pass

    def check_missing_children(self, neighbour_h,  neighbour_c):
        n_missing = self.max_output_degree - neighbour_h.size(1)
        if n_missing > 0:
            n_nodes = neighbour_h.size(0)
            h_size = neighbour_h.size(2)
            neighbour_h = th.cat((neighbour_h, self.bottom_h.reshape(1, 1, h_size).expand(n_nodes, n_missing, h_size)), dim=1)
            neighbour_c = th.cat((neighbour_c, self.bottom_c.reshape(1, 1, h_size).expand(n_nodes, n_missing, h_size)), dim=1)

        return neighbour_h, neighbour_c

    def message_func(self, edges):
        return {'h': edges.src['h'], 'c': edges.src['c']}

    def reduce_func(self, nodes):
        # add the input contribution
        neighbour_h, neighbour_c = self.check_missing_children(nodes.mailbox['h'], nodes.mailbox['c'])
        #TODO: 2 must be the max_output_degree
        f_aggr = self.f_aggregator(neighbour_h) + (nodes.data['f_input'] + self.b_f).repeat((1,2))
        iou_aggr = self.iou_aggregator(neighbour_h) + nodes.data['iou_input']

        f = th.sigmoid(f_aggr).view(*neighbour_c.size())
        c = th.sum(f * neighbour_c, 1)
        return {'iou_aggr': iou_aggr, 'c_aggr': c}

    def apply_node_func(self, nodes):
        iou = nodes.data['iou_input'] + self.b_iou
        if 'iou_aggr' in nodes.data:
            # internal nodes
            iou += nodes.data['iou_aggr']

        i, o, u = th.chunk(iou, 3, 1)
        i, o, u = th.sigmoid(i), th.sigmoid(o), th.tanh(u)
        c = i * u

        if 'c_aggr' in nodes.data:
            # internal nodes
            c += nodes.data['c_aggr']

        h = o * th.tanh(c)
        return {'h': h, 'c': c}


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

    def forward(self, g, x, mask):
        """Compute tree-lstm prediction given a batch.
        Parameters
        ----------
        batch : TreeDataset.TreeBatch
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
        g.register_message_func(self.cell.message_func)
        g.register_reduce_func(self.cell.reduce_func)
        g.register_apply_node_func(self.cell.apply_node_func)
        # feed embedding
        embeds = self.input_module(x * mask)
        g.ndata['iou_input'] = self.cell.W_iou(embeds) * mask.float().unsqueeze(-1)
        g.ndata['f_input'] = self.cell.W_f(embeds) * mask.float().unsqueeze(-1)
        # propagate
        dgl.prop_nodes_topo(g)
        # compute output
        #h = g.ndata.pop('h')
        h = g.ndata['h']
        out = self.output_module(h)
        return out


class GenericTreeRNNCell(nn.Module):

    def __init__(self, x_size, h_size, max_output_degree, cell_type, **cell_args):
        super(GenericTreeRNNCell, self).__init__()
        # TODO: add parameter to choose to freeze or not the bottom values
        self.bottom_h = nn.Parameter(th.zeros(h_size), requires_grad=False)

        # for the input
        self.max_output_degree = max_output_degree

        # TODO: 2 must be replaced by max_output_degree
        self.W_f = nn.Linear(x_size, h_size, bias=False)
        self.b_f = nn.Parameter(th.zeros(1, h_size))

        if cell_type == 'nary':
            self.h_aggregator = NaryAggregator(h_size, h_size, max_output_degree, bias=False)
        elif cell_type == 'sum':
            self.h_aggregator = SumChildAggregator()
        elif cell_type == 'hosvd':
            self.h_aggregator = HOSVDAggregator(h_size, h_size, max_output_degree, **cell_args)
        elif cell_type == 'tt':
            self.h_aggregator = TTAggregator(h_size, h_size, max_output_degree, **cell_args)
        elif cell_type == 'cd':
            self.h_aggregator = CANCOMPAggregator(h_size, h_size, max_output_degree, **cell_args)
        elif cell_type == 'full':
            if max_output_degree > 2:
                raise ValueError('Full cel type can be use only with a maximum output degree of 2')
            self.h_aggregator = BinaryFullTensorAggregator(h_size, h_size)

    def forward(self, *input):
        pass

    def check_missing_children(self, neighbour_h):
        n_missing = self.max_output_degree - neighbour_h.size(1)
        if n_missing > 0:
            n_nodes = neighbour_h.size(0)
            h_size = neighbour_h.size(2)
            neighbour_h = th.cat((neighbour_h, self.bottom_h.reshape(1, 1, h_size).expand(n_nodes, n_missing, h_size)), dim=1)

        return neighbour_h

    def message_func(self, edges):
        return {'h': edges.src['h']}

    def reduce_func(self, nodes):
        # add the input contribution
        neighbour_h = self.check_missing_children(nodes.mailbox['h'])
        h_aggr = self.h_aggregator(neighbour_h)
        return {'h': h_aggr}

    def apply_node_func(self, nodes):
        h = nodes.data['f_input'] + self.b_f
        if 'h' in nodes.data:
            # internal nodes
            h += nodes.data['h']
        h = F.tanh(h)
        return {'h': h}


class TreeRNN(nn.Module):
    def __init__(self,
                 x_size,
                 h_size,
                 max_output_degree,
                 input_module,
                 output_module,
                 cell_type='nary', **cell_args):
        super(TreeRNN, self).__init__()
        self.x_size = x_size
        self.h_size = h_size
        self.input_module = input_module
        self.output_module = output_module
        self.cell = GenericTreeRNNCell(x_size, h_size, max_output_degree, cell_type, **cell_args)

    def forward(self, g, x, mask):
        """Compute tree-lstm prediction given a batch.
        Parameters
        ----------
        batch : TreeDataset.TreeBatch
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
        g.register_message_func(self.cell.message_func)
        g.register_reduce_func(self.cell.reduce_func)
        g.register_apply_node_func(self.cell.apply_node_func)
        # feed embedding
        embeds = self.input_module(x * mask)
        g.ndata['f_input'] = self.cell.W_f(embeds) * mask.float().unsqueeze(-1)
        g.ndata['x'] = x
        # propagate
        dgl.prop_nodes_topo(g)
        # compute output
        h = g.ndata.pop('h')
        out = self.output_module(h)
        return out
