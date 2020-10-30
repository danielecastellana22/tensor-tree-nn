import numpy as np
import torch as th
from torch import nn as nn
from torch.nn import init as INIT


class BaseAggregator(nn.Module):

    # n_aggr allows to speed up the computation computing more aggregation in parallel. USEFUL FOR LSTM
    def __init__(self, h_size, pos_stationarity, max_output_degree, t_size, n_aggr):
        super(BaseAggregator, self).__init__()

        self.h_size = h_size
        self.max_output_degree = max_output_degree
        self.pos_stationarity = pos_stationarity
        self.n_aggr = n_aggr
        self.t_size = t_size

    def reset_parameters(self):
        for x in self.parameters(recurse=False):
            INIT.xavier_uniform_(x)

    # input is nieghbour_h has shape batch_size x n_neighbours x h_size
    # output has shape batch_size x (n_aggr * h_size)
    def forward(self, neighbour_h, type_embs):
        pass


# h = U1*h1 + U2*h2 + ... + Un*hn
class SumChild(BaseAggregator):

    def __init__(self, h_size, pos_stationarity=False, max_output_degree=0, t_size=None, n_aggr=1):
        super(SumChild, self).__init__(h_size, pos_stationarity, max_output_degree, t_size, n_aggr)

        if self.pos_stationarity:
            self.U = nn.Parameter(th.empty(h_size, n_aggr*h_size), requires_grad=True)
            self.b = nn.Parameter(th.empty(1, n_aggr*h_size), requires_grad=True)
        else:
            self.U = nn.Parameter(th.empty(max_output_degree*h_size, n_aggr*h_size), requires_grad=True)
            self.b = nn.Parameter(th.empty(1, n_aggr*h_size), requires_grad=True)

        if self.t_size is not None:
            self.U_type = nn.Parameter(th.empty(t_size, n_aggr * h_size), requires_grad=True)

        self.reset_parameters()

    def reset_parameters(self):
        INIT.xavier_uniform_(self.U)
        INIT.xavier_uniform_(self.b)
        if self.t_size is not None:
            INIT.xavier_uniform_(self.U_type)

    # neighbour_states has shape bs x n_ch x h
    # type_embs has shape bs x emb_s
    def forward(self, neighbour_h, type_embs=None):
        bs = neighbour_h.size(0)
        if self.pos_stationarity:
            ris = th.matmul(th.sum(neighbour_h, 1, keepdim=True), self.U).squeeze(1) + self.b
        else:
            n_ch = neighbour_h.size(1)
            ris = th.matmul(neighbour_h.view((bs, 1, -1)), self.U[:n_ch*self.h_size]).squeeze(1) + self.b[:n_ch*self.h_size]

        if self.t_size is not None:
            ris += th.matmul(type_embs, self.U_type)

        return ris


# h3 =  Canonical decomposition
class Canonical(BaseAggregator):

    def __init__(self, h_size, pos_stationarity=False, max_output_degree=0, t_size=None, n_aggr=1, rank=None):
        super(Canonical, self).__init__(h_size, pos_stationarity, max_output_degree, t_size, n_aggr)

        self.rank = rank

        if self.pos_stationarity:
            self.U = nn.Parameter(th.empty(h_size, n_aggr*rank), requires_grad=True)
            self.b = nn.Parameter(th.empty(1, n_aggr*rank), requires_grad=True)
        else:
            self.U = nn.Parameter(th.empty(max_output_degree, h_size, n_aggr * rank), requires_grad=True)
            self.b = nn.Parameter(th.empty(max_output_degree, 1, n_aggr * rank), requires_grad=True)

        self.U_output = nn.Parameter(th.empty(n_aggr, rank, h_size), requires_grad=True)
        self.b_output = nn.Parameter(th.empty(n_aggr, 1, h_size), requires_grad=True)

        if self.t_size is not None:
            self.U_type = nn.Parameter(th.empty(t_size, n_aggr * rank), requires_grad=True)
            self.b_type = nn.Parameter(th.empty(1, n_aggr * rank), requires_grad=True)

        self.reset_parameters()

    # neighbour_states has shape batch_size x n_neighbours x insize
    def forward(self, neighbour_h, type_embs=None):
        n_ch = neighbour_h.size(1)
        if not self.pos_stationarity:
            ris = th.matmul(neighbour_h.unsqueeze(2), self.U[:n_ch, :, :]) + self.b[:n_ch, :, :]
        else:
            ris = th.matmul(neighbour_h.unsqueeze(2), self.U) + self.b
        # ris has shape (bs x n_ch x 1 x n_aggr*rank)
        ris = th.prod(ris, 1)  # ris has shape (bs x 1 x n_aggr*rank)

        if self.t_size is not None:
            ris = ris * (th.matmul(type_embs, self.U_type) + self.b_type).unsqueeze(1)

        # (bs x n_aggr x 1 x rank) mul (1 x n_aggr x rank x h)
        ris = th.matmul(ris.view((-1, self.n_aggr, 1, self.rank)), self.U_output) + self.b_output
        # ris has shape (bs x n_aggr x 1 x h)
        return ris.squeeze(2).view(-1, self.n_aggr * self.h_size)


class Full  (BaseAggregator):

    def __init__(self, h_size, pos_stationarity=False, max_output_degree=0, t_size=None, n_aggr=1):
        super(Full, self).__init__(h_size, pos_stationarity, max_output_degree, t_size, n_aggr)

        in_size_list = [h_size] * max_output_degree
        if t_size is not None:
            in_size_list.insert(0, t_size)

        self.T = AugmentedTensor(in_size_list, n_aggr*h_size, pos_stationarity, n_aggr=1)

    # neighbour_states has shape batch_size x n_neighbours x insize
    def forward(self, neighbour_h, type_embs=None):
        bs = neighbour_h.size(0)
        n_ch = neighbour_h.size(1)
        #input_el = list(th.chunk(neighbour_h, self.max_output_degree, 1))
        input_el = list(th.chunk(neighbour_h, n_ch, 1))

        if type_embs is not None:
            input_el.insert(0, type_embs)

        return self.T(*input_el).view(bs, -1)


# h = U3*r3, where r3 = G*r1*r2, r1 = U1*h1 and r2 = U2*h2
class Hosvd(BaseAggregator):

    def __init__(self, h_size, pos_stationarity=False, max_output_degree=0, t_size=None, n_aggr=1,
                 rank=None, t_rank=None):
        if pos_stationarity:
            raise NotImplementedError("pos stationariy is not implemented yet!")
        super(Hosvd, self).__init__(h_size, pos_stationarity, max_output_degree, t_size, n_aggr)

        self.rank = rank
        self.t_rank = t_rank

        # mode matrices
        self.U = nn.Parameter(th.empty(max_output_degree, n_aggr, h_size, rank), requires_grad=True)
        self.b = nn.Parameter(th.empty(max_output_degree, n_aggr, 1,  rank), requires_grad=True)
        if t_size is not None:
            self.U_type = nn.Parameter(th.empty(t_size, n_aggr * self.t_rank), requires_grad=True)
            self.b_type = nn.Parameter(th.empty(1, n_aggr * self.t_rank), requires_grad=True)

        # core tensor is a fulltensor aggregator wiht r^d size
        in_size_list = [rank for i in range(max_output_degree)]
        if t_size is not None:
            in_size_list.insert(0, self.t_rank)
        self.G = AugmentedTensor(in_size_list, rank, pos_stationarity, n_aggr)

        # output matrices
        self.U_output = nn.Parameter(th.empty(n_aggr, rank, h_size), requires_grad=True)
        self.b_output = nn.Parameter(th.empty(n_aggr, 1, h_size), requires_grad=True)

        self.reset_parameters()

    # neighbour_states has shape batch_size x n_neighbours x insize
    def forward(self, neighbour_h, type_embs=None):
        n_ch = neighbour_h.size(1)
        if not self.pos_stationarity:
            ris = (th.matmul(neighbour_h.unsqueeze(2).unsqueeze(3), self.U[:n_ch, :, :]) + self.b[:n_ch, :, :]).squeeze(3)
        else:
            ris = (th.matmul(neighbour_h.unsqueeze(2).unsqueeze(3), self.U) + self.b).squeeze(3)
        # ris has shape (bs x n_ch x n_aggr x rank)
        in_el_list = []
        if self.t_size is not None:
            in_el_list.append((th.matmul(type_embs, self.U_type) + self.b_type).view(-1, self.n_aggr, self.t_rank))

        for i in range(n_ch):
            in_el_list.append(ris[:, i, :, :])

        ris = self.G(*in_el_list)  # ris has shape (bs x n_aggr x rank)
        ris = th.matmul(ris.unsqueeze(2), self.U_output) + self.b_output

        return ris.view(neighbour_h.size(0), -1)


# h3 =  tt decomposition
class TensorTrain(BaseAggregator):

    # it is weight sharing, rather than pos_stationarity
    def __init__(self, h_size, pos_stationarity=False, max_output_degree=0, t_size=None, n_aggr=1, rank=None):
        super(TensorTrain, self).__init__(h_size, pos_stationarity, max_output_degree, t_size, n_aggr)

        self.rank = rank

        if t_size is not None:
            self.U_type = nn.Parameter(th.empty(n_aggr, t_size, rank), requires_grad=True)
            self.b_type = nn.Parameter(th.empty(n_aggr, 1, rank), requires_grad=True)

        if not self.pos_stationarity:
            # mode matrices
            self.U = nn.Parameter(th.empty(max_output_degree, n_aggr, h_size, (rank+1) * rank), requires_grad=True)
            self.b = nn.Parameter(th.empty(max_output_degree, n_aggr, 1, (rank + 1) * rank), requires_grad=True)
        else:
            self.U = nn.Parameter(th.empty(n_aggr, h_size, (rank + 1) * rank), requires_grad=True)
            self.b = nn.Parameter(th.empty(n_aggr, 1, (rank + 1) * rank), requires_grad=True)

        # output matrices
        self.U_output = nn.Parameter(th.empty(n_aggr, rank, h_size), requires_grad=True)
        self.b_output = nn.Parameter(th.empty(n_aggr, 1, h_size), requires_grad=True)

        self.reset_parameters()

    # neighbour_states has shape batch_size x n_neighbours x insize
    def forward(self, neighbour_h, type_embs=None):
        bs = neighbour_h.size(0)
        n_ch = neighbour_h.size(1)

        if not self.pos_stationarity:
            ris = (th.matmul(neighbour_h.unsqueeze(2).unsqueeze(3), self.U[:n_ch, :, :, :]) + self.b[:n_ch, :, :, :]).squeeze(3)
        else:
            ris = (th.matmul(neighbour_h.unsqueeze(2).unsqueeze(3), self.U) + self.b).squeeze(3)
        # ris has shape bs x n_ch x n_aggr x (r+1)*r
        rank_tens_list = th.chunk(ris, n_ch, 1)

        if type_embs is not None:
            rank_ris = th.matmul(type_embs.unsqueeze(1).unsqueeze(2), self.U_type) + self.b_type
            # has shape bs x n_agg x 1 x t_rank
        else:
            rank_ris = None

        # multiply by the rank along the chain
        for rank_tens in rank_tens_list:
            aux = rank_tens.view(bs, self.n_aggr, self.rank+1, self.rank)
            U_rank = aux[:,:, :-1,:]
            b_rank = aux[:,:, -2:-1,:]

            if rank_ris is None:
                rank_ris = b_rank
            else:
                rank_ris = th.matmul(rank_ris, U_rank) + b_rank

        out = th.matmul(rank_ris, self.U_output) + self.b_output # has shape bs x n_agg x 1 x h_size

        return out.view(neighbour_h.size(0), -1)


# TODO: deal with n_ch < max_output_degree when pos_stationarity is False
# h3 =  tt decomposition
class TensorTrainLMN(BaseAggregator):

    # it is weight sharing, rather than pos_stationarity
    def __init__(self, h_size, pos_stationarity=False, max_output_degree=0, t_size=None, n_aggr=1, rank=None):
        super(TensorTrainLMN, self).__init__(h_size, pos_stationarity, max_output_degree, t_size, n_aggr)

        self.rank = rank

        if t_size is not None:
            self.U_type = nn.Parameter(th.empty(n_aggr, t_size, rank), requires_grad=True)
            self.b_type = nn.Parameter(th.empty(n_aggr, 1, rank), requires_grad=True)

        if not self.pos_stationarity:
            # mode matrices
            self.A = nn.Parameter(th.empty(max_output_degree, n_aggr, h_size, rank), requires_grad=True)
            self.B = nn.Parameter(th.empty(max_output_degree, n_aggr, rank, rank), requires_grad=True)
            self.A_b = nn.Parameter(th.empty(max_output_degree, n_aggr, 1, rank), requires_grad=True)
        else:
            self.A = nn.Parameter(th.empty(n_aggr, h_size, rank), requires_grad=True)
            self.B = nn.Parameter(th.empty(n_aggr, rank, rank), requires_grad=True)
            self.A_b = nn.Parameter(th.empty(n_aggr, 1, rank), requires_grad=True)

        # output matrices
        self.U_output = nn.Parameter(th.empty(n_aggr, rank, h_size), requires_grad=True)
        self.b_output = nn.Parameter(th.empty(n_aggr, 1, h_size), requires_grad=True)

        self.reset_parameters()

    # neighbour_states has shape batch_size x n_neighbours x insize
    def forward(self, neighbour_h, type_embs=None):
        bs = neighbour_h.size(0)
        n_ch = neighbour_h.size(1)

        ris = (th.matmul(neighbour_h.unsqueeze(2).unsqueeze(3), self.A) + self.A_b).squeeze(3)
        # ris has shape bs x n_ch x n_aggr x r
        ax_list = th.chunk(ris, n_ch, 1)

        if type_embs is not None:
            rank_ris = th.matmul(type_embs.unsqueeze(1).unsqueeze(2), self.U_type) + self.b_type
            # has shape bs x n_agg x 1 x rank
        else:
            rank_ris = None

        # multiply by the rank along the chain
        for ax_el in ax_list:

            if rank_ris is None:
                rank_ris = ax_el.squeeze(1).unsqueeze(2)
            else:
                rank_ris = th.matmul(rank_ris, self.B) + ax_el.squeeze(1).unsqueeze(2)

        out = th.matmul(rank_ris, self.U_output) + self.b_output # has shape bs x n_agg x 1 x h_size

        return out.view(neighbour_h.size(0), -1)


class AugmentedTensor(nn.Module):

    def __init__(self, in_size_list, out_size, pos_stationarity, n_aggr):

        if n_aggr * np.prod(in_size_list) * out_size > 10**9:
            raise ValueError('Too many parameters!')

        super(AugmentedTensor, self).__init__()

        self.n_aggr = n_aggr
        # +1 for the bias
        self.in_size_list = [x+1 for x in in_size_list]
        self.n_input = len(in_size_list)
        self.out_size = out_size
        self.pos_stationarity = pos_stationarity

        if self.pos_stationarity:
            raise NotImplementedError('Full with stationarity not implemented yet')
        else:
            d = self.in_size_list
            d.append(out_size)
            # the n_aggr for batching slow down everything
            self.T_list = nn.ParameterList()
            for i in range(self.n_aggr):
                self.T_list.append(nn.Parameter(th.empty(*d), requires_grad=True))

        self.reset_parameters()

    def reset_parameters(self):
        for t in self.T_list:
            INIT.xavier_uniform_(t)
            #INIT.orthogonal_(t)

    # neighbour_states has shape batch_size x n_neighbours x insize
    def forward(self, *in_el_list):
        bs = in_el_list[0].size(0)
        n_aggr = self.n_aggr

        ris_list = []
        h = th.cat((in_el_list[0].view(bs, n_aggr, -1), th.ones((bs, n_aggr, 1), device=in_el_list[0].device)), dim=2)
        for j in range(n_aggr):
            ris_list.append(th.matmul(h[:, j, :], self.T_list[j].view(self.in_size_list[0], -1)))

        for i in range(1, self.n_input):
            dim_i = self.in_size_list[i]
            if i < len(in_el_list):
                in_el = in_el_list[i] # has shape (bs x n_aggr x h)
                h = th.cat((in_el, th.ones((bs, n_aggr, 1), device=in_el.device)), dim=2)
            else:
                in_el = th.zeros((bs, n_aggr, dim_i), device=in_el_list[0].device)
                in_el[:, :, -1] = -1

            for j in range(n_aggr):
                ris_list[j] = th.bmm(h[:, j:j+1, :], ris_list[j].view(bs,  dim_i, -1))

        return th.cat(ris_list, dim=1)