import torch as th
import torch.nn as nn

# TODO: the rank is / 3 due to the three gate
# TODO: maybe is reasonable a flag for positional stationarity
class GenericTreeLSTMCell(nn.Module):

    def __init__(self, h_size, max_output_degree):
        super(GenericTreeLSTMCell, self).__init__()

        self.h_size = h_size
        self.max_output_degree = max_output_degree

        # TODO: add parameter to choose to freeze or not the bottom values. Tensor or grad=False?
        self.bottom_h = nn.Parameter(th.zeros(h_size), requires_grad=False)
        self.bottom_c = nn.Parameter(th.zeros(h_size), requires_grad=False)

        # matrix for the forget gate. WE ASSUME FORGET GATE DEPENDS ONLY ON h_i
        self.U_f = nn.Linear(h_size, h_size, bias=False)

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

    def compute_forget_gate(self, neighbour_h):
        return self.U_f(neighbour_h.view(-1, self.h_size)).view(-1, self.max_output_degree * self.h_size)

    def compute_iou_gate(self, neighbour_h):
        raise NotImplementedError('users must define compute_iou to use this base class')

    def message_func(self, edges):
        return {'h': edges.src['h'], 'c': edges.src['c']}

    def reduce_func(self, nodes):
        #check missin child
        neighbour_h, neighbour_c = self.check_missing_children(nodes.mailbox['h'], nodes.mailbox['c'])

        # add the input contribution
        f_aggr = self.compute_forget_gate(neighbour_h) + (nodes.data['f_input']).repeat((1, self.max_output_degree))
        iou_aggr = self.compute_iou_gate(neighbour_h) + nodes.data['iou_input']

        f = th.sigmoid(f_aggr).view(*neighbour_c.size())
        c = th.sum(f * neighbour_c, 1)
        return {'iou_aggr': iou_aggr, 'c_aggr': c}

    def apply_node_func(self, nodes):
        iou = nodes.data['iou_input']
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


# The Full aggregator is available only when maxOutput Degree is 2
# h = A*h1*h2 + U1*h1 + U2*h2 + b
class BinaryFullTensorCell(GenericTreeLSTMCell):

    def __init__(self, h_size, max_output_degree):
        if max_output_degree > 2:
            raise ValueError('Full cel type can be use only with a maximum output degree of 2')

        super(BinaryFullTensorCell, self).__init__(h_size, max_output_degree)

        # define parameters for the computation of iou
        self.A = nn.Parameter(th.rand(h_size, h_size, 3*h_size))
        self.U1 = nn.Linear(h_size, 3*h_size, bias=False)
        self.U2 = nn.Linear(h_size, 3*h_size, bias=True)

    # neighbour_states has shape batch_size x n_neighbours x insize
    def compute_iou_gate(self, neighbour_h):
        h1 = neighbour_h[:, 0, :].view(neighbour_h.size(0), -1)
        h2 = neighbour_h[:, 1, :].view(neighbour_h.size(0), -1)
        return th.einsum('ijk,ni,nj->nk', self.A, h1, h2) + self.U1(h1) + self.U2(h2)


# h = U1*h1 + U2*h2 + ... + Un*hn
class NaryCell(GenericTreeLSTMCell):

    def __init__(self, h_size, max_output_degree):
        super(NaryCell, self).__init__(h_size, max_output_degree)

        # define parameters for the computation of iou
        self.U = nn.Linear(max_output_degree * h_size, 3*h_size, bias=False)

    # neighbour_states has shape batch_size x n_neighbours x insize
    def compute_iou_gate(self, neighbour_h):
        # no missing children
        h_cat = neighbour_h.view(neighbour_h.size(0), -1)
        return self.U(h_cat)


# h = h1 + h2 + ... + hn
class SumChildCell(GenericTreeLSTMCell):

    def __init__(self, h_size, max_output_degree):
        super(SumChildCell, self).__init__(h_size, max_output_degree)
        self.U = nn.Linear(h_size, 3*h_size, bias=False)

    # neighbour_states has shape batch_size x n_neighbours x insize
    def compute_iou_gate(self, neighbour_h):
        return self.U(th.sum(neighbour_h, 1))


# TODO: what about bias
# h = U3*r3, where r3 = G*r1*r2, r1 = U1*h1 and r2 = U2*h2
class HOSVDCell(GenericTreeLSTMCell):

    def __init__(self, h_size, max_output_degree, rank):
        super(HOSVDCell, self).__init__(h_size, max_output_degree)

        self.rank = rank

        # core tensor
        sz_G = tuple(rank for i in range(max_output_degree+1))
        self.G = nn.Parameter(th.rand(sz_G))

        # mode matrices
        self.U_list = nn.ParameterList()
        for i in range(max_output_degree):
            self.U_list.append(nn.Parameter(th.rand((h_size, rank))))

        self.U_output = nn.Parameter(th.rand((rank, 3*h_size)))

    # neighbour_states has shape batch_size x n_neighbours x insize
    def compute_iou_gate(self, neighbour_h):
        # first pos
        h = neighbour_h[:, 0, :].view(neighbour_h.size(0), -1)
        U = self.U_list[0]
        # size is N \times r^d
        r_out = th.chain_matmul(h, U, self.G.view(self.rank, -1))

        for i in range(1, self.max_output_degree):
            h = neighbour_h[:, i, :].view(neighbour_h.size(0), -1)
            U = self.U_list[i]
            ris = th.matmul(h, U)

            r_out = th.bmm(r_out.view(h.size()[0], -1, self.rank), ris.view(-1, self.rank, 1))

        h_out = th.matmul(r_out.squeeze(), self.U_output)

        return h_out


# h3 =  Canonical decomposition
class CANCOMPCell(GenericTreeLSTMCell):

    def __init__(self, h_size, max_output_degree, rank):
        super(CANCOMPCell, self).__init__(h_size, max_output_degree)

        self.rank = rank

        # mode matrices
        self.U_list = nn.ParameterList()
        for i in range(max_output_degree):
            self.U_list.append(nn.Parameter(th.rand((h_size, rank))))

        self.U_output = nn.Parameter(th.rand((rank, 3*h_size)))

    # neighbour_states has shape batch_size x n_neighbours x insize
    def compute_iou_gate(self, neighbour_h):
        for i in range(self.max_output_degree):
            h = neighbour_h[:, i, :].view(neighbour_h.size(0), -1)
            U = self.U_list[i]
            if i == 0:
                ris = th.matmul(h, U)
            else:
                ris = ris * th.matmul(h, U)

        h_out = th.matmul(ris, self.U_output)

        return h_out


class TTCell(GenericTreeLSTMCell):

    def __init__(self, h_size, max_output_degree, rank):
        super(TTCell, self).__init__(h_size, max_output_degree)

        self.rank = rank

        self.U_fisrt = nn.Parameter(th.rand((h_size, rank)))

        # TT tensors
        self.U_list = nn.ParameterList()
        for i in range(max_output_degree):
            self.U_list.append(nn.Parameter(th.rand((h_size, rank * rank))))

        self.U_last = nn.Parameter(th.rand((rank, 3*h_size)))

    # neighbour_states has shape batch_size x n_neighbours x insize
    def compute_iou_gate(self, neighbour_h):
        h = neighbour_h[:, 0, :].view(neighbour_h.size(0), -1)
        ris = th.matmul(h, self.U_fisrt).view(-1, self.rank, 1)

        for i in range(1, self.max_output_degree):
            h = neighbour_h[:, i, :].view(neighbour_h.size(0), -1)
            U = self.U_list[i]
            ris = th.bmm(th.matmul(h, U).view((-1, self.rank, self.rank)), ris)

        h_out = th.matmul(ris.squeeze(), self.U_last)
        return h_out
