import torch.nn as nn
import torch as th
import numpy as np


class BaseAggregator(nn.Module):

    # n_aggr allows to speed up the computation computing more aggregation in parallel. USEFUL FOR LSTM
    def __init__(self, h_size, max_output_degree, pos_stationarity, n_aggr):
        super(BaseAggregator, self).__init__()

        self.n_aggr = n_aggr
        self.h_size = h_size
        self.max_output_degree = max_output_degree
        self.pos_stationarity = pos_stationarity

    # input is nieghbour_h has shape batch_size x n_neighbours x h_size
    # output has shape batch_size x (n_aggr * h_size)
    def forward(self, neighbour_h, nodes):
        pass


class AugmentedTensor(nn.Module):

    def __init__(self, in_size_list, out_size, pos_stationarity):

        if np.prod(in_size_list) > 10**9:
            raise ValueError('Too many parameters!')

        super(AugmentedTensor, self).__init__()
        self.in_size_list = in_size_list
        self.max_output_degree = len(in_size_list)
        self.out_size = out_size
        self.pos_stationarity = pos_stationarity

        if self.pos_stationarity:
            #TODO: does not work. Output must be 3*h_size
            raise NotImplementedError('Full with stationarity not implemented yet')
        else:
            # +1 for the bias
            d = [x+1 for x in self.in_size_list]
            d.insert(1, out_size)
            self.A = nn.Parameter(th.randn(*d))

    # neighbour_states has shape batch_size x n_neighbours x insize
    def forward(self, n_batch, *in_el_list):
        if self.pos_stationarity:
            raise NotImplementedError('Full with stationarity not implemented yet')
        else:
            A = self.A

        ris = None
        for i in range(self.max_output_degree):
            in_el = in_el_list[i]
            if ris is None:
                # add bias. h has size N_BATCH x H_SIZE+1
                h = th.cat((in_el.view(n_batch, -1), th.ones((n_batch, 1), device=in_el.device)), dim=1)
                ris = th.matmul(h, A.view((self.in_size_list[i]+1, -1)))
            else:
                # add bias. h has size N_BATCH x H_SIZE+1 x 1
                h = th.cat((in_el.view(n_batch, -1, 1), th.ones((n_batch, 1, 1), device=in_el.device)), dim=1)
                ris = th.bmm(ris.view((n_batch, -1, self.in_size_list[i]+1)), h)

        return ris.squeeze(dim=2)


class FullTensor(BaseAggregator):

    def __init__(self, h_size, max_output_degree, pos_stationarity, n_aggr, **kwargs):
        if h_size**max_output_degree > 10**9:
            raise ValueError('Too many parameters!')

        super(FullTensor, self).__init__(h_size, max_output_degree, pos_stationarity, n_aggr)

        in_size_list = [h_size for i in range(max_output_degree)]
        out_size = n_aggr*h_size
        self.T = AugmentedTensor(in_size_list, out_size, pos_stationarity)

    # neighbour_states has shape batch_size x n_neighbours x insize
    def forward(self, neighbour_h, nodes):

        n_batch = neighbour_h.size(0)
        return self.T(n_batch, *th.chunk(neighbour_h, self.max_output_degree, 1))


# h = U1*h1 + U2*h2 + ... + Un*hn
class SumChild(BaseAggregator):

    def __init__(self, h_size, max_output_degree, pos_stationarity, n_aggr, **kwargs):
        super(SumChild, self).__init__(h_size, max_output_degree, pos_stationarity, n_aggr)

        if not self.pos_stationarity:
            self.U = nn.Linear(self.max_output_degree * self.h_size, self.n_aggr*self.h_size, bias=True)
        else:
            self.U = nn.Linear(self.h_size, self.n_aggr * self.h_size, bias=True)

    # neighbour_states has shape batch_size x n_neighbours x insize
    def forward(self, neighbour_h, nodes):
        if not self.pos_stationarity:
            return self.U(neighbour_h.view(neighbour_h.size(0), -1))
        else:
            return self.U(th.sum(neighbour_h, 1))


# h = U3*r3, where r3 = G*r1*r2, r1 = U1*h1 and r2 = U2*h2
class Hosvd(BaseAggregator):

    def __init__(self, h_size, max_output_degree, pos_stationarity, n_aggr, **kwargs):
        super(Hosvd, self).__init__(h_size, max_output_degree, pos_stationarity, n_aggr)

        rank = kwargs['rank']
        self.rank = rank

        # core tensor is a fulltensor aggregator wiht r^d size
        self.G_list = nn.ModuleList()
        in_size_list = [rank for i in range(max_output_degree)]
        out_size = rank
        for i in range(n_aggr):
            self.G_list.append(AugmentedTensor(in_size_list, out_size, pos_stationarity))

        if self.pos_stationarity:
            raise NotImplementedError("pos stationariy is not implemented yet!")
        else:
            # mode matrices
            self.U_list = nn.ModuleList()
            for i in range(max_output_degree):
                self.U_list.append(nn.Linear(h_size, n_aggr*rank, bias=True))

        # output matrices
        self.U_output_list = nn.ModuleList()
        for i in range(n_aggr):
            self.U_output_list.append(nn.Linear(rank, h_size))

    # neighbour_states has shape batch_size x n_neighbours x insize
    def forward(self, neighbour_h, nodes):
        rank_list = []
        if self.pos_stationarity:
            raise NotImplementedError("pos stationariy is not implemented yet!")
        else:
            # obtain a tensor n_batch x n_neighbours x rank x n_aggr to pass to the full aggregator
            for i in range(self.max_output_degree):
                h_i = th.squeeze(neighbour_h[:, i, :], 1)
                rank_list.append(th.chunk(self.U_list[i](h_i).reshape(-1, self.rank, self.n_aggr), self.n_aggr, dim=2))

            n_batch = neighbour_h.size(0)
            ris = None
            for i in range(self.n_aggr):
                in_el_list = [x[i] for x in rank_list]
                if ris is None:
                    ris = self.U_output_list[i](self.G_list[i](n_batch, *in_el_list))
                else:
                    aux = self.U_output_list[i](self.G_list[i](n_batch, *in_el_list))
                    ris = th.cat((ris, aux), dim=1)

        return ris


# h3 =  Canonical decomposition
class Canonical(BaseAggregator):

    def __init__(self, h_size, max_output_degree, pos_stationarity, n_aggr, **kwargs):
        super(Canonical, self).__init__(h_size, max_output_degree, pos_stationarity, n_aggr)

        rank = kwargs['rank']
        self.rank = rank

        if not self.pos_stationarity:
            # mode matrices
            self.U_list = nn.ModuleList()
            for i in range(max_output_degree):
                self.U_list.append(nn.Linear(h_size, n_aggr*rank, bias=True))
        else:
            self.U = nn.Linear(h_size, n_aggr * rank, bias=True)

        # output matrices
        self.U_output_list = nn.ModuleList()
        for i in range(n_aggr):
            self.U_output_list.append(nn.Linear(rank, h_size))

    # neighbour_states has shape batch_size x n_neighbours x insize
    def forward(self, neighbour_h, nodes):
        ris = None
        for i in range(self.max_output_degree):
            h = neighbour_h[:, i, :].view(neighbour_h.size(0), -1)

            #assume bottom = 1 => speed up computation
            #if th.sum(h) == 0:
            #   continue

            if not self.pos_stationarity:
                U = self.U_list[i]
            else:
                U = self.U

            if ris is None:
                ris = U(h)
            else:
                ris = ris * U(h)

        in_list = th.chunk(ris, self.n_aggr, 1)
        out_tens = None

        for i in range(self.n_aggr):
            r_in = in_list[i]
            r_out = self.U_output_list[i](r_in)

            if out_tens is None:
                out_tens = r_out
            else:
                out_tens = th.cat((out_tens, r_out), dim=1)

        return out_tens


# h3 =  Canonical decomposition
class TensorTrain(BaseAggregator):

    def __init__(self, h_size, max_output_degree, pos_stationarity, n_aggr, **kwargs):
        super(TensorTrain, self).__init__(h_size, max_output_degree, pos_stationarity, n_aggr)

        rank = kwargs['rank']
        self.rank = rank

        if not self.pos_stationarity:
            # mode matrices
            self.U_list = nn.ModuleList()

            self.U_list.append(nn.Linear(h_size, n_aggr*rank, bias=True))
            in_size_list = [h_size, rank]
            for i in range(max_output_degree-1):
                self.U_list.append(nn.Linear(h_size, n_aggr * (rank+1) * rank, bias=True))
                #self.U_list.append(AugmentedTensor(in_size_list, n_aggr*rank, pos_stationarity))
        else:
            # probaby is not possible
            raise NotImplementedError("pos stationariy is not implemented yet!")

        # output matrices
        self.U_output_list = nn.ModuleList()
        for i in range(n_aggr):
            self.U_output_list.append(nn.Linear(rank, h_size, bias=True))

    # neighbour_states has shape batch_size x n_neighbours x insize
    def forward(self, neighbour_h, nodes):

        n_batch = neighbour_h.size(0)
        rank_mat_list = []
        # multiply by the hidden state
        for i in range(self.max_output_degree):
            h = neighbour_h[:, i, :].view(n_batch, -1)

            if not self.pos_stationarity:
                U = self.U_list[i]
            else:
                raise NotImplementedError("pos stationariy is not implemented yet!")

            if i == 0:
                rank_mat_list.append(U(h).reshape((n_batch * self.n_aggr, self.rank, 1)))
            else:
                rank_mat_list.append(U(h).reshape((n_batch * self.n_aggr, self.rank, self.rank+1)))

        # multiply by the rank along the chain
        rank_ris = rank_mat_list[0]
        for i in range(1, self.max_output_degree):
            rank_ris = th.cat((rank_ris, th.ones((n_batch*self.n_aggr, 1, 1), device=rank_ris.device)), dim=1)
            aux = rank_mat_list[i]
            rank_ris = th.bmm(aux, rank_ris)

        rank_ris = rank_ris.reshape((n_batch, self.n_aggr, self.rank))
        in_list = th.chunk(rank_ris, self.n_aggr, 1)
        out_tens = None

        for i in range(self.n_aggr):
            r_in = in_list[i]
            r_out = self.U_output_list[i](r_in).reshape((-1, self.h_size))

            if out_tens is None:
                out_tens = r_out
            else:
                out_tens = th.cat((out_tens, r_out), dim=1)

        return out_tens

# use different parameter for each type node
class TypedAggregator(BaseAggregator):

    def __init__(self, h_size, max_output_degree, pos_stationarity, n_aggr, **kwargs):
        super(TypedAggregator, self).__init__(h_size, max_output_degree, pos_stationarity, n_aggr)

        self.n_type = kwargs['n_type']
        self.cell_list = nn.ModuleList()
        for i in range(self.n_type):
            self.cell_list.append(kwargs['agg_class'](h_size, max_output_degree, pos_stationarity, n_aggr, **kwargs))

    def forward(self, neighbour_h, nodes):

        # get type
        ris = th.zeros((neighbour_h.size(0), self.n_aggr*neighbour_h.size(2)), device=neighbour_h.device)
        for i in range(self.n_type):
            mask = nodes.data['type'] == i
            if th.sum(mask) > 0:
                ris[mask, :] = self.cell_list[i](neighbour_h[mask, :, :], nodes)

        return ris
