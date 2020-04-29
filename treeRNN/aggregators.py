import torch.nn as nn
import torch as th
import torch.nn.init as INIT
from .utils import AugmentedTensor


class BaseAggregator(nn.Module):

    # n_aggr allows to speed up the computation computing more aggregation in parallel. USEFUL FOR LSTM
    def __init__(self, h_size, max_output_degree, pos_stationarity, n_aggr, type_emb_size):
        super(BaseAggregator, self).__init__()

        self.h_size = h_size
        self.max_output_degree = max_output_degree
        self.pos_stationarity = pos_stationarity
        self.n_aggr = n_aggr
        self.type_emb_size = type_emb_size

    def reset_parameters(self):
        pass

    # input is nieghbour_h has shape batch_size x n_neighbours x h_size
    # output has shape batch_size x (n_aggr * h_size)
    def forward(self, neighbour_h, type_embs):
        pass


# h = U1*h1 + U2*h2 + ... + Un*hn
class SumChild(BaseAggregator):

    def __init__(self, h_size, max_output_degree, pos_stationarity, n_aggr, type_emb_size=None):
        super(SumChild, self).__init__(h_size, max_output_degree, pos_stationarity, n_aggr, type_emb_size)

        if self.pos_stationarity:
            dim_U = [h_size, n_aggr*h_size]
            dim_b = [1, n_aggr*h_size]
        else:
            dim_U = [max_output_degree*h_size, n_aggr*h_size]
            dim_b = [1, n_aggr*h_size]

        self.U = nn.Parameter(th.Tensor(*dim_U), requires_grad=True)
        self.b = nn.Parameter(th.Tensor(*dim_b), requires_grad=True)

        if self.type_emb_size is not None:
            self.U_type = nn.Parameter(th.Tensor(type_emb_size, n_aggr * h_size), requires_grad=True)
            #self.b_type = nn.Parameter(th.Tensor(1, n_aggr * h_size), requires_grad=True)

        self.reset_parameters()

    def reset_parameters(self):
        INIT.xavier_uniform_(self.U)
        INIT.xavier_uniform_(self.b)
        if self.type_emb_size is not None:
            INIT.xavier_uniform_(self.U_type)
            #INIT.xavier_uniform_(self.b_type)

    # neighbour_states has shape bs x n_ch x h
    # type_embs has shape bs x emb_s
    def forward(self, neighbour_h, type_embs=None):
        n_aggr = self.n_aggr
        h = self.h_size
        n_ch = self.max_output_degree
        bs = neighbour_h.size(0)
        emb_s = self.type_emb_size

        U = self.U
        b = self.b
        if self.pos_stationarity:
            ris = th.matmul(th.sum(neighbour_h, 1, keepdim=True), U).squeeze(1) + b
        else:
            ris = th.matmul(neighbour_h.view((bs, 1, -1)), U).squeeze(1) + b

        if self.type_emb_size is not None:
            ris += th.matmul(type_embs, self.U_type) #+ self.b_type)

        return ris


# h3 =  Canonical decomposition
class Canonical(BaseAggregator):

    def __init__(self, h_size, max_output_degree, pos_stationarity, n_aggr, rank, type_emb_size=None):
        super(Canonical, self).__init__(h_size, max_output_degree, pos_stationarity, n_aggr, type_emb_size)
        self.rank = rank

        if self.pos_stationarity:
            dim_U = [h_size, n_aggr*rank]
            dim_b = [1, n_aggr*rank]
            dim_U_out = [n_aggr, rank, h_size]
            dim_b_out = [n_aggr, 1, h_size]
        else:
            dim_U = [max_output_degree, h_size, n_aggr * rank]
            dim_b = [max_output_degree, 1, n_aggr * rank]
            dim_U_out = [n_aggr, rank, h_size]
            dim_b_out = [n_aggr, 1, h_size]

        self.U = nn.Parameter(th.randn(*dim_U), requires_grad=True)
        self.b = nn.Parameter(th.randn(*dim_b), requires_grad=True)
        self.U_output = nn.Parameter(th.randn(*dim_U_out), requires_grad=True)
        self.b_output = nn.Parameter(th.randn(*dim_b_out), requires_grad=True)

        if self.type_emb_size is not None:
            self.U_type = nn.Parameter(th.Tensor(type_emb_size, n_aggr * rank), requires_grad=True)
            self.b_type = nn.Parameter(th.Tensor(1, n_aggr * rank), requires_grad=True)

        self.reset_parameters()

    def reset_parameters(self):
        INIT.xavier_uniform_(self.U)
        INIT.xavier_uniform_(self.b)
        INIT.xavier_uniform_(self.U_output)
        INIT.xavier_uniform_(self.b_output)

        if self.type_emb_size is not None:
            INIT.xavier_uniform_(self.U_type)
            INIT.xavier_uniform_(self.b_type)

    # neighbour_states has shape batch_size x n_neighbours x insize
    def forward(self, neighbour_h, type_embs=None):

        bs = neighbour_h.size(0)
        h = self.h_size
        n_ch = neighbour_h.size(1)  # self.max_output_degree
        n_aggr = self.n_aggr
        rank = self.rank
        emb_s = self.type_emb_size

        U = self.U
        b = self.b
        U_out = self.U_output
        b_out = self.b_output
        # neighbour_h has shape (bs x n_ch x 1 x h)
        # U has shape (1 x h x n_agr*rank)
        ris = th.matmul(neighbour_h.view((bs, n_ch, 1, h)), U) + b  # ris has shape (bs x n_ch x 1 x n_aggr*rank)
        ris = th.prod(ris, 1)  # ris has shape (bs x 1 x n_aggr*rank)

        if self.type_emb_size is not None:
            ris = ris * (th.matmul(type_embs, self.U_type) + self.b_type).view(bs, 1, -1)

        # (bs x n_aggr x 1 x rank) mul (1 x n_aggr x rank x h)
        ris = th.matmul(ris.view((bs, n_aggr, 1, rank)), U_out) + b_out
        # ris has shape (bs x n_aggr x 1 x h)
        return ris.squeeze(2).view(-1, n_aggr*h)


class FullTensor(BaseAggregator):

    def __init__(self, h_size, max_output_degree, pos_stationarity, n_aggr, type_emb_size=None):
        if h_size**max_output_degree > 10**9:
            raise ValueError('Too many parameters!')

        if pos_stationarity:
            raise ValueError('Full tensor cannot be pos stationary!')

        super(FullTensor, self).__init__(h_size, max_output_degree, pos_stationarity, n_aggr, type_emb_size)

        in_size_list = [h_size for i in range(max_output_degree)]
        if type_emb_size is not None:
            in_size_list.insert(0, type_emb_size)

        self.T = AugmentedTensor(1, in_size_list, n_aggr*h_size, pos_stationarity)

    # neighbour_states has shape batch_size x n_neighbours x insize
    def forward(self, neighbour_h, type_embs=None):
        bs = neighbour_h.size(0)
        input_el = list(th.chunk(neighbour_h, self.max_output_degree, 1))

        if type_embs is not None:
            input_el.insert(0, type_embs)

        return self.T(*input_el).view(bs, -1)


# h = U3*r3, where r3 = G*r1*r2, r1 = U1*h1 and r2 = U2*h2
class Hosvd(BaseAggregator):

    def __init__(self, h_size, max_output_degree, pos_stationarity, n_aggr, rank, type_emb_size=None, type_emb_rank=None):
        if pos_stationarity:
            raise NotImplementedError("pos stationariy is not implemented yet!")

        super(Hosvd, self).__init__(h_size, max_output_degree, pos_stationarity, n_aggr, type_emb_size)

        self.rank = rank
        self.type_embs_rank = type_emb_rank

        if self.pos_stationarity:
            raise NotImplementedError("Pos stationarity is not implemented yet!")

        dim_U = [max_output_degree, n_aggr, h_size, rank]
        dim_b = [max_output_degree, n_aggr, 1,  rank]
        dim_U_out = [n_aggr, rank, h_size]
        dim_b_out = [n_aggr, 1, h_size]

        if self.type_emb_size is not None:
            self.U_type = nn.Parameter(th.Tensor(type_emb_size, n_aggr * self.type_embs_rank), requires_grad=True)
            self.b_type = nn.Parameter(th.Tensor(1, n_aggr * self.type_embs_rank), requires_grad=True)

        # core tensor is a fulltensor aggregator wiht r^d size
        in_size_list = [rank for i in range(max_output_degree)]
        if type_emb_size is not None:
            in_size_list.insert(0, self.type_embs_rank)
        self.G = AugmentedTensor(n_aggr, in_size_list, rank, pos_stationarity)

        # mode matrices
        self.U = nn.Parameter(th.Tensor(*dim_U), requires_grad=True)
        self.b = nn.Parameter(th.Tensor(*dim_b), requires_grad=True)

        # output matrices
        self.U_output = nn.Parameter(th.Tensor(*dim_U_out), requires_grad=True)
        self.b_output = nn.Parameter(th.Tensor(*dim_b_out), requires_grad=True)

        self.reset_parameters()

    def reset_parameters(self):
        INIT.xavier_uniform_(self.U)
        INIT.xavier_uniform_(self.b)
        INIT.xavier_uniform_(self.U_output)
        INIT.xavier_uniform_(self.b_output)

        if self.type_emb_size is not None:
            INIT.xavier_uniform_(self.U_type)
            INIT.xavier_uniform_(self.b_type)

    # neighbour_states has shape batch_size x n_neighbours x insize
    def forward(self, neighbour_h, type_embs=None):
        bs = neighbour_h.size(0)
        h = self.h_size
        n_ch = self.max_output_degree
        n_aggr = self.n_aggr
        rank = self.rank
        emb_s = self.type_emb_size

        if self.pos_stationarity:
            raise NotImplementedError("Pos stationariy is not implemented yet!")
            # TODO: should be applied only on true childs. Missing children should be considered separately.

        ris = (th.matmul(neighbour_h.view((bs, n_ch, 1, 1, h)), self.U) + self.b).squeeze(3)
        # ris has shape (bs x n_ch x n_aggr x rank)
        in_el_list = []
        if self.type_emb_size is not None:
            in_el_list.append((th.matmul(type_embs, self.U_type) + self.b_type).view(-1, n_aggr, self.type_embs_rank))

        for i in range(self.max_output_degree):
            in_el_list.append(ris[:, i, :, :])

        ris = self.G(*in_el_list)  # ris has shape (bs x n_aggr x rank)
        ris = th.matmul(ris.unsqueeze(2), self.U_output) + self.b_output

        return ris.view(bs, -1)


# h3 =  tt decomposition
class TensorTrain(BaseAggregator):

    # it is weight sharing, rather than pos_stationarity
    def __init__(self, h_size, max_output_degree, pos_stationarity, n_aggr, rank):
        super(TensorTrain, self).__init__(h_size, max_output_degree, pos_stationarity, n_aggr)

        raise NotImplementedError('TT not implemented yet')
        self.rank = rank

        if not self.pos_stationarity:
            # mode matrices
            self.U_list = nn.ModuleList()

            for i in range(max_output_degree):
                self.U_list.append(nn.Linear(h_size, n_aggr * (rank+1) * rank, bias=True))
        else:
            self.U = nn.Linear(h_size, n_aggr * (rank+1) * rank, bias=True)

        # output matrices
        self.U_output_list = nn.ModuleList()
        for i in range(n_aggr):
            self.U_output_list.append(nn.Linear(rank, h_size, bias=True))

    # neighbour_states has shape batch_size x n_neighbours x insize
    def forward(self, neighbour_h):

        n_batch = neighbour_h.size(0)
        n_ch = neighbour_h.size(1)
        rank_mat_list = []
        # multiply by the hidden state
        for i in range(n_ch):
            h = neighbour_h[:, i, :].view(n_batch, -1)

            if not self.pos_stationarity:
                U = self.U_list[i]
            else:
                U = self.U

            rank_mat_list.append(U(h).reshape((n_batch * self.n_aggr, self.rank, self.rank+1)))

        # multiply by the rank along the chain
        rank_ris = th.zeros((n_batch * self.n_aggr, self.rank, 1), device=neighbour_h.device)
        for i in range(n_ch):
            rank_ris = th.cat((rank_ris, th.ones((n_batch*self.n_aggr, 1, 1), device=rank_ris.device)), dim=1)
            aux = rank_mat_list[i]
            rank_ris = nn.functional.tanh(th.bmm(aux, rank_ris))
            #rank_ris = th.bmm(aux, rank_ris)

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


class GRUAggregator(BaseAggregator):

    def __init__(self, h_size, max_output_degree, pos_stationarity, n_aggr, rank):
        super(GRUAggregator, self).__init__(h_size, max_output_degree, pos_stationarity, n_aggr)

        self.rank = rank

        if not self.pos_stationarity:
            raise ValueError('GRUAggregator is stationarity!')
        else:
            self.U_output_list = nn.ModuleList()
            self.GRU_list = nn.ModuleList()
            for i in range(n_aggr):
                self.U_output_list.append(nn.Linear(rank, h_size, bias=True))
                self.GRU_list.append(nn.GRU(input_size=h_size, hidden_size=rank, num_layers=1, bias=True, batch_first=True))

    # neighbour_states has shape batch_size x n_neighbours x insize
    def forward(self, neighbour_h):

        n_batch = neighbour_h.size(0)
        n_ch = neighbour_h.size(1)
        rank_mat_list = []

        out_list = []
        for i in range(self.n_aggr):
            gru = self.GRU_list[i]
            U_out = self.U_output_list[i]
            gru_out = gru(neighbour_h)[1].view(-1, self.rank)
            out_list.append(U_out(gru_out))

        return th.cat(out_list, dim=1)


class HierarchicalAggregator(BaseAggregator):

    def __init__(self, h_size, max_output_degree, pos_stationarity, n_aggr, max_n_el, aggr_class, rank=None):
        super(HierarchicalAggregator, self).__init__(h_size, max_output_degree, pos_stationarity, n_aggr)
        self.max_n_el = max_n_el
        self.aggr_module_list = nn.ModuleList()
        for i in range(n_aggr):
            if rank:
                self.aggr_module_list.append(aggr_class(h_size, max_n_el, pos_stationarity, n_aggr=1, rank=rank))
            else:
                self.aggr_module_list.append(aggr_class(h_size, max_n_el, pos_stationarity, n_aggr=1))

    def forward(self, neighbour_h):
        n_batch = neighbour_h.size(0)
        x = neighbour_h

        n_ch = x.size(1)
        if n_ch > self.max_n_el:
            if n_ch % self.max_n_el > 0:
                x = th.cat([x, th.zeros((n_batch, self.max_n_el - n_ch % self.max_n_el, self.h_size), device=x.device)], 1)
            n_blocchi = x.size(1) // self.max_n_el
            x = x.view((-1, self.max_n_el, self.h_size))
        else:
            n_blocchi = 1

        in_ist = [x for i in range(self.n_aggr)]
        while n_blocchi > 1:
            for i in range(self.n_aggr):
                x = self.aggr_module_list[i](in_ist[i]).view(-1, n_blocchi, self.h_size)

                n_ch = x.size(1)
                if n_ch > self.max_n_el:
                    if n_ch % self.max_n_el > 0:
                        x = th.cat([x, th.zeros((n_batch, self.max_n_el - n_ch % self.max_n_el, self.h_size),
                                                device=x.device)], 1)
                    n_blocchi_new = x.size(1) // self.max_n_el
                    x = x.view((-1, self.max_n_el, self.h_size))
                else:
                    n_blocchi_new = 1

                in_ist[i] = x
            n_blocchi = n_blocchi_new

        out_list = []
        for i in range(self.n_aggr):
            out_list.append(self.aggr_module_list[i](in_ist[i]))

        ris = th.cat(out_list, 1)
        assert ris.size(0) == n_batch
        return ris