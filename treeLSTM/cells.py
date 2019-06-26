import torch as th
import torch.nn as nn
import math
import numpy as np

from itertools import permutations

def __bin_coeff__(n,k):
    return math.factorial(n) // (math.factorial(k) * math.factorial(n-k))


def __comb_with_rep__(n,k):
    return __bin_coeff__(n+k-1, k)


# we assume size contains same len, i.e otuptu has size n^d \time n.
# only n^d is symmentric.
def __get_symmetric_idx__(n, d):
    out_shape = tuple([n for i in range(d+1)])
    n_el = n**(d+1)
    v_idx = {}
    gather_idx = np.array([-1 for i in range(n_el)])

    for i in range(n_el):
        i_tuple = list(np.unravel_index(i, out_shape))
        to_sort_idx = i_tuple[:-1]
        last_idx = i_tuple[-1]
        i_tuple_sorted = tuple(sorted(to_sort_idx) + [last_idx])

        if i_tuple_sorted in v_idx:
            idx_param = v_idx[i_tuple_sorted]
        else:
            idx_param = len(v_idx)
            v_idx[i_tuple_sorted] = idx_param

        gather_idx[i] = idx_param

    #TODO: return a tensor and register as a buffer
    return nn.Parameter(th.LongTensor(gather_idx), requires_grad=False)


def __get_symmetric_tensor_view__(w, idx, out_shape, output_axis):
    #TODO: this function can be called only after a optim.step() rahter than every time
    return w.gather(0, idx).view(out_shape).transpose(output_axis, len(out_shape)-1).contiguous()


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
            #self.U_f = nn.Linear(h_size, h_size, bias=False)
            self.U_f = nn.Parameter(th.randn(h_size, h_size))
        else:
            self.U_f_list = nn.ParameterList()

            for i in range(max_output_degree):
                self.U_f_list.append(nn.Parameter(th.randn(h_size, h_size)))

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
            return th.matmul(neighbour_h.view(-1, self.h_size), self.U_f).view(-1, self.max_output_degree * self.h_size)
        else:
            ris = None
            for i in range(self.max_output_degree):
                U = self.U_f_list[i]
                h = neighbour_h[:, i, :].view(-1, self.h_size)
                if ris is not None:
                    ris = th.cat((ris, th.matmul(h, U)), dim=1)
                else:
                    ris = th.matmul(h, U)

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


# The Full aggregator is available only when maxOutput Degree is 2
# h = A*h1*h2 + U1*h1 + U2*h2 + b
class BinaryFullTensorCell(GenericTreeLSTMCell):

    def __init__(self, h_size, max_output_degree, pos_stationarity):
        if max_output_degree > 2:
            raise ValueError('Full cel type can be use only with a maximum output degree of 2')

        super(BinaryFullTensorCell, self).__init__(h_size, max_output_degree, pos_stationarity)

        self.h_size = h_size
        self.max_output_degree = max_output_degree

        #TODO: use parameter instead of module
        if self.pos_stationarity:
            n_diff_elements = __comb_with_rep__(h_size, 2)
            self.A_w = nn.Parameter(th.randn(n_diff_elements * h_size))
            self.symmetric_idx = __get_symmetric_idx__(h_size, 2)
            self.U = nn.Linear(h_size, 3*h_size, bias=False)
        else:
            # define parameters for the computation of iou

            self.A = nn.Parameter(th.rand(h_size, h_size, 3*h_size))
            self.U1 = nn.Linear(h_size, 3*h_size, bias=False)
            self.U2 = nn.Linear(h_size, 3*h_size, bias=True)

    # neighbour_states has shape batch_size x n_neighbours x insize
    def compute_iou_gate(self, neighbour_h):
        if self.pos_stationarity:
            A = __get_symmetric_tensor_view__(self.A_w, self.symmetric_idx, (self.h_size, self.h_size, self.h_size), 2)
            U1 = self.U
            U2 = self.U
        else:
            A = self.A
            U1 = self.U1
            U2 = self.U2
        h1 = neighbour_h[:, 0, :].view(neighbour_h.size(0), -1)
        h2 = neighbour_h[:, 1, :].view(neighbour_h.size(0), -1)
        return th.einsum('ijk,ni,nj->nk', A, h1, h2) + U1(h1) + U2(h2)


# h = U1*h1 + U2*h2 + ... + Un*hn
class NaryCell(GenericTreeLSTMCell):

    def __init__(self, h_size, max_output_degree, pos_stationarity):
        super(NaryCell, self).__init__(h_size, max_output_degree, pos_stationarity)
        #TODO: use parameter instead of module
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


# h = U3*r3, where r3 = G*r1*r2, r1 = U1*h1 and r2 = U2*h2
class HOSVDCell(GenericTreeLSTMCell):

    def __init__(self, h_size, max_output_degree, rank, pos_stationarity):
        super(HOSVDCell, self).__init__(h_size, max_output_degree, pos_stationarity)

        self.rank = rank
        sz_G = tuple(rank for i in range(max_output_degree + 1))
        self.G_size = sz_G

        if not self.pos_stationarity:

            sz_G = tuple(rank for i in range(max_output_degree + 1))
            # core tensor
            self.G_i = nn.Parameter(th.randn(sz_G))
            self.G_o = nn.Parameter(th.randn(sz_G))
            self.G_u = nn.Parameter(th.randn(sz_G))

            # mode matrices
            self.U_list = nn.ParameterList()
            self.B_list = nn.ParameterList()
            for i in range(max_output_degree):
                self.U_list.append(nn.Parameter(th.randn((h_size, 3*rank))))
                self.B_list.append(nn.Parameter(th.randn(3*rank)))
        else:
            # symmetric core tensor
            n_diff_elements = __comb_with_rep__(rank, max_output_degree)
            self.G_i_w = nn.Parameter(th.randn(n_diff_elements * rank))
            self.G_o_w = nn.Parameter(th.randn(n_diff_elements * rank))
            self.G_u_w = nn.Parameter(th.randn(n_diff_elements * rank))

            self.symmetric_idx = __get_symmetric_idx__(rank, max_output_degree)

            # shared mode matrices
            self.U = nn.Parameter(th.randn((h_size, 3*rank)))
            self.B = nn.Parameter(th.randn(3*rank))

        # output matrices
        self.Ui_output = nn.Parameter(th.randn((rank, h_size)))
        self.Uo_output = nn.Parameter(th.randn((rank, h_size)))
        self.Uu_output = nn.Parameter(th.randn((rank, h_size)))

    # neighbour_states has shape batch_size x n_neighbours x insize
    def compute_iou_gate(self, neighbour_h):
        if not self.pos_stationarity:
            # first pos
            h = neighbour_h[:, 0, :].view(neighbour_h.size(0), -1)
            U = self.U_list[0]
            b = self.B_list[0]
            # size is N \times r^d
            aux_i, aux_o, aux_u = th.addmm(b, h, U).chunk(3, dim=1)
            r_i = th.matmul(aux_i, self.G_i.view(self.rank, -1))
            r_o = th.matmul(aux_o, self.G_o.view(self.rank, -1))
            r_u = th.matmul(aux_u, self.G_u.view(self.rank, -1))

            for i in range(1, self.max_output_degree):
                h = neighbour_h[:, i, :].view(neighbour_h.size(0), -1)
                U = self.U_list[i]
                b = self.B_list[i]
                aux_i, aux_o, aux_u = th.addmm(b, h, U).chunk(3, dim=1)
                r_i = th.bmm(r_i.view(h.size()[0], -1, self.rank), aux_i.view(-1, self.rank, 1))
                r_o = th.bmm(r_o.view(h.size()[0], -1, self.rank), aux_o.view(-1, self.rank, 1))
                r_u = th.bmm(r_u.view(h.size()[0], -1, self.rank), aux_u.view(-1, self.rank, 1))

        else:
            aux = th.addmm(self.B, neighbour_h.view((-1, self.h_size)), self.U).view(-1, self.max_output_degree, 3*self.rank)
            aux_i, aux_o, aux_u = aux[:, 0, :].squeeze(1).chunk(3, dim=1)

            G_i = __get_symmetric_tensor_view__(self.G_i_w, self.symmetric_idx, self.G_size, 1)
            G_o = __get_symmetric_tensor_view__(self.G_o_w, self.symmetric_idx, self.G_size, 1)
            G_u = __get_symmetric_tensor_view__(self.G_u_w, self.symmetric_idx, self.G_size, 1)

            r_i = th.matmul(aux_i, G_i.view(self.rank, -1))
            r_o = th.matmul(aux_o, G_o.view(self.rank, -1))
            r_u = th.matmul(aux_u, G_u.view(self.rank, -1))

            for i in range(1, self.max_output_degree):
                aux_i, aux_o, aux_u = aux[:, i, :].squeeze(1).chunk(3, dim=1)
                r_i = th.bmm(r_i.view(aux.size()[0], -1, self.rank), aux_i.view(-1, self.rank, 1))
                r_o = th.bmm(r_o.view(aux.size()[0], -1, self.rank), aux_o.view(-1, self.rank, 1))
                r_u = th.bmm(r_u.view(aux.size()[0], -1, self.rank), aux_u.view(-1, self.rank, 1))

        gate_i = th.matmul(r_i.squeeze(2), self.Ui_output)
        gate_o = th.matmul(r_o.squeeze(2), self.Uo_output)
        gate_u = th.matmul(r_u.squeeze(2), self.Uu_output)

        return th.cat((gate_i, gate_o, gate_u), dim=1)


# h3 =  Canonical decomposition
class CANCOMPCell(GenericTreeLSTMCell):

    def __init__(self, h_size, max_output_degree, rank, pos_stationarity):
        super(CANCOMPCell, self).__init__(h_size, max_output_degree, pos_stationarity)

        self.rank = rank

        if not self.pos_stationarity:
            # mode matrices
            self.U_list = nn.ParameterList()
            self.B_list = nn.ParameterList()
            for i in range(max_output_degree):
                self.U_list.append(nn.Parameter(th.randn((h_size, 3*rank))))
                self.B_list.append(nn.Parameter(th.randn(3*rank)))
        else:
            # mode matrices shared
            self.U = nn.Parameter(th.randn((h_size, 3*rank)))
            self.B = nn.Parameter(th.randn(3*rank))

        # TODO: maybe is better using 3 x rank x hsize with batched matrix multiplication
        self.Ui_output = nn.Parameter(th.randn((rank, h_size)))
        self.Uo_output = nn.Parameter(th.randn((rank, h_size)))
        self.Uu_output = nn.Parameter(th.randn((rank, h_size)))

    # neighbour_states has shape batch_size x n_neighbours x insize
    def compute_iou_gate(self, neighbour_h):
        for i in range(self.max_output_degree):
            h = neighbour_h[:, i, :].view(neighbour_h.size(0), -1)
            if not self.pos_stationarity:
                U = self.U_list[i]
                B = self.B_list[i]
            else:
                U = self.U
                B = self.B

            if i == 0:
                ris = th.addmm(B, h, U)
            else:
                ris = ris * th.addmm(B, h, U)

        (r_i, r_o, r_u) = th.chunk(ris, 3, 1)
        gate_i = th.matmul(r_i, self.Ui_output)
        gate_o = th.matmul(r_o, self.Uo_output)
        gate_u = th.matmul(r_u, self.Uu_output)

        return th.cat((gate_i, gate_o, gate_u), dim=1)


# TODO: what about pos stationarity?
# PROBABLY, tensor Train can't express positional stationarity
class TTCell(GenericTreeLSTMCell):

    def __init__(self, h_size, max_output_degree, rank, pos_stationarity):
        super(TTCell, self).__init__(h_size, max_output_degree, pos_stationarity)

        if pos_stationarity:
            raise NotImplementedError('TT with positional stationarity is not implemented yet.')
        else:
            self.rank = rank

            self.U_fisrt = nn.Parameter(th.randn((h_size, 3*rank)))
            self.B_first = nn.Parameter(th.randn(3 * rank))

            # TT tensors
            self.Ui_list = nn.ParameterList()
            self.Bi_list = nn.ParameterList()

            self.Uo_list = nn.ParameterList()
            self.Bo_list = nn.ParameterList()

            self.Uu_list = nn.ParameterList()
            self.Bu_list = nn.ParameterList()
            for i in range(max_output_degree-1):
                self.Ui_list.append(nn.Parameter(th.randn((h_size, rank*rank))))
                self.Bi_list.append(nn.Parameter(th.randn(rank)))

                self.Uo_list.append(nn.Parameter(th.randn((h_size, rank*rank))))
                self.Bo_list.append(nn.Parameter(th.randn(rank)))

                self.Uu_list.append(nn.Parameter(th.randn((h_size, rank*rank))))
                self.Bu_list.append(nn.Parameter(th.randn(rank)))

            self.Ui_output = nn.Parameter(th.randn((rank, h_size)))
            self.Uo_output = nn.Parameter(th.randn((rank, h_size)))
            self.Uu_output = nn.Parameter(th.randn((rank, h_size)))

    # neighbour_states has shape batch_size x n_neighbours x insize
    def compute_iou_gate(self, neighbour_h):

        if self.pos_stationarity:
            raise NotImplementedError('TT with positional stationarity is not implemented yet.')
        else:
            h = neighbour_h[:, 0, :].view(neighbour_h.size(0), -1)

            ris = th.addmm(self.B_first, h, self.U_fisrt).view(-1, 3*self.rank, 1)
            (r_i, r_o, r_u) = th.chunk(ris, 3, 1)
            for i in range(0, self.max_output_degree-1):
                h = neighbour_h[:, i+1, :].view(neighbour_h.size(0), -1)

                U_i = self.Ui_list[i]
                B_i = self.Bi_list[i].view(1, self.rank, 1).expand(neighbour_h.size(0), -1, -1)

                U_o = self.Uo_list[i]
                B_o = self.Bo_list[i].view(1, self.rank, 1).expand(neighbour_h.size(0), -1, -1)

                U_u = self.Uu_list[i]
                B_u = self.Bu_list[i].view(1, self.rank, 1).expand(neighbour_h.size(0), -1, -1)

                r_i = th.baddbmm(B_i, th.matmul(h, U_i).view(-1, self.rank, self.rank), r_i)
                r_o = th.baddbmm(B_o, th.matmul(h, U_o).view(-1, self.rank, self.rank), r_i)
                r_u = th.baddbmm(B_u, th.matmul(h, U_u).view(-1, self.rank, self.rank), r_i)

            gate_i = th.matmul(r_i.squeeze(2), self.Ui_output)
            gate_o = th.matmul(r_o.squeeze(2), self.Uo_output)
            gate_u = th.matmul(r_u.squeeze(2), self.Uu_output)

            return th.cat((gate_i, gate_o, gate_u), dim=1)
