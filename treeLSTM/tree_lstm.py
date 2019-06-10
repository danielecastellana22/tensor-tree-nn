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
    def __init__(self, in_size, out_size, max_output_degree):
        super(NaryAggregator, self).__init__()
        self.max_output_degree = max_output_degree
        self.U = nn.Linear(max_output_degree * in_size, out_size, bias=False)

    # neighbour_states has shape batch_size x n_neighbours x insize
    def forward(self, neighbour_states):
        # no missing children
        h_cat = neighbour_states.view(neighbour_states.size(0), -1)
        return self.U(h_cat)


# h = h1 + h2 + ... + hn
class SumChildAggregator(nn.Module):

    def __init__(self, in_size, out_size):
        super(SumChildAggregator, self).__init__()
        self.U = nn.Linear(in_size, out_size, bias=False)

    # neighbour_states has shape batch_size x n_neighbours x insize
    def forward(self, neighbour_states):
        return self.U(th.sum(neighbour_states, 1))


class SumChildAggregator_F(nn.Module):

    def __init__(self, in_size, out_size):
        super(SumChildAggregator_F, self).__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.U = nn.Linear(in_size, out_size, bias=False)

    # neighbour_states has shape batch_size x n_neighbours x insize
    def forward(self, neighbour_states):
        return self.U(neighbour_states.view(-1, self.in_size)).view(-1, neighbour_states.size()[1] * self.out_size)


# TODO: what about bias
# h = U3*r3, where r3 = G*r1*r2, r1 = U1*h1 and r2 = U2*h2
class HOSVDAggregator(nn.Module):
    def __init__(self, in_size, out_size, max_output_degree, rank):
        super(HOSVDAggregator, self).__init__()

        self.max_output_degree = max_output_degree
        self.in_size = in_size
        self.out_size = out_size
        self.rank = rank

        # core tensor
        sz_G = tuple(rank for i in range(max_output_degree+1))
        self.G = nn.Parameter(th.rand(sz_G))

        # mode matrices
        self.U_list = nn.ParameterList()
        for i in range(max_output_degree):
            self.U_list.append(nn.Parameter(th.rand((in_size, rank))))

        self.U_output = nn.Parameter(th.rand((rank, out_size)))

    # neighbour_states has shape batch_size x n_neighbours x insize
    def forward(self, neighbour_states):
        # first pos
        h = neighbour_states[:, 0, :].view(neighbour_states.size(0), -1)
        U = self.U_list[0]
        # size is N \times r^d
        r_out = th.chain_matmul(h, U, self.G.view(self.rank, -1))

        for i in range(1, self.max_output_degree):
            h = neighbour_states[:, i, :].view(neighbour_states.size(0), -1)
            U = self.U_list[i]
            ris = th.matmul(h, U)

            r_out = th.bmm(r_out.view(h.size()[0], -1, self.rank), ris.view(-1, self.rank, 1))

        h_out = th.matmul(r_out.squeeze(), self.U_output)

        return h_out


# h3 =  Canonical decomposition
class CANCOMPAggregator(nn.Module):
    def __init__(self, in_size, out_size, max_output_degree, rank):
        super(CANCOMPAggregator, self).__init__()

        # CANCOMP matrices
        self.max_output_degree = max_output_degree

        # mode matrices
        self.U_list = nn.ParameterList()
        for i in range(max_output_degree):
            self.U_list.append(nn.Parameter(th.rand((in_size, rank))))

        self.U_output = nn.Parameter(th.rand((rank, out_size)))

    # neighbour_states has shape batch_size x n_neighbours x insize
    def forward(self, neighbour_states):
        for i in range(self.max_output_degree):
            h = neighbour_states[:, i, :].view(neighbour_states.size(0), -1)
            U = self.U_list[i]
            if i ==0:
                ris = th.matmul(h, U)
            else:
                ris = ris * th.matmul(h, U)

        h_out = th.matmul(ris, self.U_output)

        return h_out


class TTAggregator(nn.Module):
    def __init__(self, in_size, out_size, max_output_degree, rank):
        super(TTAggregator, self).__init__()

        self.max_output_degree = max_output_degree
        self.in_size = in_size
        self.out_size = out_size
        self.rank = rank

        self.U_fisrt = nn.Parameter(th.rand((in_size, rank)))

        # TT tensors
        self.U_list = nn.ParameterList()
        for i in range(max_output_degree):
            self.U_list.append(nn.Parameter(th.rand((in_size, rank * rank))))

        self.U_last = nn.Parameter(th.rand((rank, out_size)))

    # neighbour_states has shape batch_size x n_neighbours x insize
    def forward(self, neighbour_states):

        h = neighbour_states[:, 0, :].view(neighbour_states.size(0), -1)
        ris = th.matmul(h, self.U_fisrt).view(-1, self.rank, 1)

        for i in range(1, self.max_output_degree):
            h = neighbour_states[:, i, :].view(neighbour_states.size(0), -1)
            U = self.U_list[i]
            ris = th.bmm(th.matmul(h, U).view((-1, self.rank, self.rank)), ris)

        h_out = th.matmul(ris.squeeze(), self.U_last)
        return h_out


class GenericTreeLSTMCell(nn.Module):

    def __init__(self, x_size, h_size, max_output_degree, cell_type, **cell_args):
        super(GenericTreeLSTMCell, self).__init__()
        # TODO: add parameter to choose to freeze or not the bottom values. Tensor or grad=False?
        self.bottom_h = nn.Parameter(th.zeros(h_size), requires_grad=False)
        self.bottom_c = nn.Parameter(th.zeros(h_size), requires_grad=False)

        # for the input
        self.max_output_degree = max_output_degree

        self.W_iou = nn.Linear(x_size, 3 * h_size, bias=False)
        self.b_iou = nn.Parameter(th.zeros(1, 3 * h_size))

        self.W_f = nn.Linear(x_size, h_size, bias=False)
        self.b_f = nn.Parameter(th.zeros(1, h_size))

        # one forget gate for each child
        # 3 remains 3 fot i, o, u
        if cell_type == 'nary':
            self.f_aggregator = NaryAggregator(h_size, max_output_degree*h_size, max_output_degree)
            self.iou_aggregator = NaryAggregator(h_size, 3*h_size, max_output_degree)
        elif cell_type == 'sum':
            self.f_aggregator = SumChildAggregator_F(h_size, h_size)
            #self.f_aggregator = SumChildAggregator(h_size, h_size*max_output_degree)
            self.iou_aggregator = SumChildAggregator(h_size, 3*h_size)
        elif cell_type == 'hosvd':
            self.f_aggregator = HOSVDAggregator(h_size, max_output_degree*h_size, max_output_degree, **cell_args)
            self.iou_aggregator = HOSVDAggregator(h_size, 3*h_size, max_output_degree, **cell_args)
        elif cell_type == 'tt':
            self.f_aggregator = TTAggregator(h_size, max_output_degree*h_size, max_output_degree, **cell_args)
            self.iou_aggregator = TTAggregator(h_size, 3*h_size, max_output_degree, **cell_args)
        elif cell_type == 'cd':
            self.f_aggregator = CANCOMPAggregator(h_size, max_output_degree*h_size, max_output_degree, **cell_args)
            self.iou_aggregator = CANCOMPAggregator(h_size, 3*h_size, max_output_degree, **cell_args)
        elif cell_type == 'full':
            if max_output_degree > 2:
                raise ValueError('Full cel type can be use only with a maximum output degree of 2')
            self.f_aggregator = BinaryFullTensorAggregator(h_size, max_output_degree*h_size)
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
        #check missin child
        neighbour_h, neighbour_c = self.check_missing_children(nodes.mailbox['h'], nodes.mailbox['c'])

        # add the input contribution
        f_aggr = self.f_aggregator(neighbour_h) + (nodes.data['f_input'] + self.b_f).repeat((1, self.max_output_degree))
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
        # TODO: x * mask does not make sense. use x[mask]
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
