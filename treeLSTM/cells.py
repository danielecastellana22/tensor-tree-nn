import torch as th
import torch.nn as nn

from itertools import permutations

def __create_symmetric_tensor__(size, output_axis):
    # TODO: implement this function!!
    aa =3


class GenericTreeLSTMCell(nn.Module):

    def __init__(self, h_size, max_output_degree, pos_stationarity):
        super(GenericTreeLSTMCell, self).__init__()

        self.h_size = h_size
        self.max_output_degree = max_output_degree
        self.pos_stationarity = pos_stationarity

        # TODO: add parameter to choose to freeze or not the bottom values. Tensor or grad=False?
        self.bottom_h = nn.Parameter(th.zeros(h_size), requires_grad=False)
        self.bottom_c = nn.Parameter(th.zeros(h_size), requires_grad=False)

        # matrix for the forget gate. WE ASSUME FORGET GATE DEPENDS ONLY ON h_i
        if pos_stationarity:
            self.U_f = nn.Linear(h_size, h_size, bias=False)
        else:
            self.U_f_list = nn.ParameterList()

            for i in range(max_output_degree):
                # TODO: must be a parameter
                self.U_f_list.append(nn.Linear(h_size, h_size, bias=False))

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
        if self.pos_stationarity:
            return self.U_f(neighbour_h.view(-1, self.h_size)).view(-1, self.max_output_degree * self.h_size)
        else:
            ris = None
            for i in range(self.max_output_degree):
                U = self.U_f_list[i]
                h = neighbour_h[:, i, :].view(-1, self.h_size)
                if ris:
                    ris = th.cat((ris, U(h)), dim=1)
                else:
                    ris = U(h)

            return ris

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


# TODO: impement one aggrefator for each gate!
# The Full aggregator is available only when maxOutput Degree is 2
# h = A*h1*h2 + U1*h1 + U2*h2 + b
class BinaryFullTensorCell(GenericTreeLSTMCell):

    def __init__(self, h_size, max_output_degree, pos_stationarity):
        if max_output_degree > 2:
            raise ValueError('Full cel type can be use only with a maximum output degree of 2')

        super(BinaryFullTensorCell, self).__init__(h_size, max_output_degree, pos_stationarity)

        if self.pos_stationarity:
            # TODO: implement positionality
            raise NotImplementedError('This is not implemented yet')
        else:
            # define parameters for the computation of iou
            self.A = nn.Parameter(th.rand(h_size, h_size, 3*h_size))
            self.U1 = nn.Linear(h_size, 3*h_size, bias=False)
            self.U2 = nn.Linear(h_size, 3*h_size, bias=True)

    # neighbour_states has shape batch_size x n_neighbours x insize
    def compute_iou_gate(self, neighbour_h):
        if self.pos_stationarity:
            # TODO: implement positionality
            raise NotImplementedError('This is not implemented yet')
        else:
            h1 = neighbour_h[:, 0, :].view(neighbour_h.size(0), -1)
            h2 = neighbour_h[:, 1, :].view(neighbour_h.size(0), -1)
            return th.einsum('ijk,ni,nj->nk', self.A, h1, h2) + self.U1(h1) + self.U2(h2)


# h = U1*h1 + U2*h2 + ... + Un*hn
class NaryCell(GenericTreeLSTMCell):

    def __init__(self, h_size, max_output_degree, pos_stationarity=True):
        super(NaryCell, self).__init__(h_size, max_output_degree, pos_stationarity)

        if not self.pos_stationarity:
            #NARY aggregation
            # define parameters for the computation of iou
            self.U = nn.Linear(max_output_degree * h_size, 3*h_size, bias=False)
        else:
            #SUMCHILD aggrefation
            self.U = nn.Linear(h_size, 3 * h_size, bias=False)

    # neighbour_states has shape batch_size x n_neighbours x insize
    def compute_iou_gate(self, neighbour_h):
        if not self.pos_stationarity:
            return self.U(neighbour_h.view(neighbour_h.size(0), -1))
        else:
            return self.U(th.sum(neighbour_h, 1))


# TODO: what about bias
# h = U3*r3, where r3 = G*r1*r2, r1 = U1*h1 and r2 = U2*h2
class HOSVDCell(GenericTreeLSTMCell):

    def __init__(self, h_size, max_output_degree, rank, pos_stationarity=True):
        super(HOSVDCell, self).__init__(h_size, max_output_degree, pos_stationarity)

        self.rank = rank

        if not self.pos_stationarity:
            # core tensor
            sz_G = tuple(rank for i in range(max_output_degree + 1))
            self.G_i = nn.Parameter(th.rand(sz_G))
            self.G_o = nn.Parameter(th.rand(sz_G))
            self.G_u = nn.Parameter(th.rand(sz_G))

            # mode matrices
            self.U_list = nn.ParameterList()
            for i in range(max_output_degree):
                self.U_list.append(nn.Parameter(th.rand((h_size, 3*rank))))
        else:
            raise NotImplementedError('This is not implemented yet')
            # TODO: core tensors must be symmetric
            # symmetric core tensor
            sz_G = tuple(rank for i in range(max_output_degree + 1))
            self.G = __create_symmetric_tensor__(sz_G, 1)

            # shared mode matrices
            self.U = nn.Parameter(th.rand((h_size, rank)))

        self.Ui_output = nn.Parameter(th.rand((rank, h_size)))
        self.Uo_output = nn.Parameter(th.rand((rank, h_size)))
        self.Uu_output = nn.Parameter(th.rand((rank, h_size)))

    # neighbour_states has shape batch_size x n_neighbours x insize
    def compute_iou_gate(self, neighbour_h):
        # first pos
        h = neighbour_h[:, 0, :].view(neighbour_h.size(0), -1)
        if not self.pos_stationarity:
            U = self.U_list[0]
        else:
            U = self.U
        # size is N \times r^d
        aux_i, aux_o, aux_u = th.matmul(h, U).chunk(3, dim=1)
        r_i = th.matmul(aux_i, self.G_i.view(self.rank, -1))
        r_o = th.matmul(aux_o, self.G_o.view(self.rank, -1))
        r_u = th.matmul(aux_u, self.G_u.view(self.rank, -1))

        for i in range(1, self.max_output_degree):
            h = neighbour_h[:, i, :].view(neighbour_h.size(0), -1)
            U = self.U_list[i]
            aux_i, aux_o, aux_u = th.matmul(h, U).chunk(3, dim=1)
            r_i = th.bmm(r_i.view(h.size()[0], -1, self.rank), aux_i.view(-1, self.rank, 1))
            r_o = th.bmm(r_o.view(h.size()[0], -1, self.rank), aux_o.view(-1, self.rank, 1))
            r_u = th.bmm(r_u.view(h.size()[0], -1, self.rank), aux_u.view(-1, self.rank, 1))

        gate_i = th.matmul(r_i, self.Ui_output)
        gate_o = th.matmul(r_o, self.Uo_output)
        gate_u = th.matmul(r_u, self.Uu_output)

        return th.cat((gate_i, gate_o, gate_u), dim=1)


# h3 =  Canonical decomposition
class CANCOMPCell(GenericTreeLSTMCell):

    def __init__(self, h_size, max_output_degree, rank, pos_stationarity=True):
        super(CANCOMPCell, self).__init__(h_size, max_output_degree, pos_stationarity)

        self.rank = rank

        if not self.pos_stationarity:
            # mode matrices
            self.U_list = nn.ParameterList()
            for i in range(max_output_degree):
                self.U_list.append(nn.Parameter(th.rand((h_size, 3*rank))))

        else:
            # mode matrices shared
            self.U = nn.Parameter(th.rand((h_size, 3*rank)))

        self.Ui_output = nn.Parameter(th.rand((rank, h_size)))
        self.Uo_output = nn.Parameter(th.rand((rank, h_size)))
        self.Uu_output = nn.Parameter(th.rand((rank, h_size)))

    # neighbour_states has shape batch_size x n_neighbours x insize
    def compute_iou_gate(self, neighbour_h):
        for i in range(self.max_output_degree):
            h = neighbour_h[:, i, :].view(neighbour_h.size(0), -1)
            if not self.pos_stationarity:
                U = self.U_list[i]
            else:
                U = self.U
            if i == 0:
                ris = th.matmul(h, U)
            else:
                ris = ris * th.matmul(h, U)

        (r_i, r_o, r_u) = th.chunk(ris, 3, 1)
        gate_i = th.matmul(r_i, self.Ui_output)
        gate_o = th.matmul(r_o, self.Uo_output)
        gate_u = th.matmul(r_u, self.Uu_output)

        return th.cat((gate_i, gate_o, gate_u), dim=1)


# Tensor Train can't express positional stationarity
class TTCell(GenericTreeLSTMCell):

    # TODO: one rank for each gate
    # TODO: what about pos stationarity?
    def __init__(self, h_size, max_output_degree, rank):
        super(TTCell, self).__init__(h_size, max_output_degree, pos_stationarity=False)

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
